#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import multiprocessing
import os
import random

# import some common libraries
import cv2
import numpy as np
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
import torch

import utils.io
import utils.segmentation


# Setup detectron2 logger
setup_logger()


def convert_bbox_mode(annot, fn_prefix=None):
    for ann in annot['annotations']:
        ann['bbox_mode'] = BoxMode(ann['bbox_mode'])
    if fn_prefix is not None:
        annot['file_name'] = os.path.join(fn_prefix, annot['file_name'])
    return annot


def get_rat_dict(annot_fn=None, fn_prefix=None):
    if annot_fn is None:
        return []

    with open(annot_fn, 'r') as f:
        data = json.load(f)
    data = list(map(lambda d: convert_bbox_mode(d, fn_prefix=fn_prefix), data))
    return data


def load_and_register_dataset(db_name, db_annot, is_part=False, fn_prefix=None):
    DatasetCatalog.register(db_name, (lambda d=db_name: get_rat_dict(db_annot, fn_prefix=fn_prefix)))
    if is_part:
        MetadataCatalog.get(db_name).set(thing_classes=['head', 'tail_base', 'tail'])
    else:
        MetadataCatalog.get(db_name).set(thing_classes=['rat'], keypoint_names=['head', 'tail_base', 'tail'],
                                         keypoint_flip_map=[('head', 'tail')],
                                         keypoint_connection_rules=[('head', 'tail_base', (255, 0, 0)),
                                                                    ('tail_base', 'tail', (0, 255, 0))])


def load_and_register_datasets(train_annot, test_annot=None, is_part=False):
    if test_annot is None:
        test_annot = train_annot
    load_and_register_dataset('rat_train', train_annot, is_part=is_part)
    load_and_register_dataset('rat_test', test_annot, is_part=is_part)


def set_model_configuration(out_dir, n_iter=10000, lr=0.00025, is_part=False, inst_only=False,
                            model_dir=None, model_ckpt=None, cpu_only=False):
    if model_ckpt is None:
        model_ckpt = "model_final"
    if model_dir is None:
        model_dir = out_dir
    print("[Info]", model_ckpt, os.path.isfile(os.path.join(model_dir, "{}.pth".format(model_ckpt))),
          os.path.join(model_dir, "{}.pth".format(model_ckpt)))

    cfg = get_cfg()
    if is_part:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        if os.path.isfile(os.path.join(model_dir, "{}.pth".format(model_ckpt))):
            cfg.MODEL.WEIGHTS = os.path.join(model_dir, "{}.pth".format(model_ckpt))
        else:
            print('cfg:', model_ckpt)
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        if not inst_only:
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        if os.path.isfile(os.path.join(model_dir, "{}.pth".format(model_ckpt))):
            cfg.MODEL.WEIGHTS = os.path.join(model_dir, "{}.pth".format(model_ckpt))
        else:
            print('cfg:', model_ckpt)
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        if not inst_only:
            cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 3
            cfg.TEST.KEYPOINT_OKS_SIGMAS = [.079, .107, .089]  # [head ~ shoulders, tail_base ~ hips, tail ~ ankles]
    cfg.DATASETS.TRAIN = ('rat_train',)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = n_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.OUTPUT_DIR = out_dir
    cfg.SOLVER.MAX_ITER = 100000

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.DATASETS.TEST = ('rat_test',)

    if cpu_only:
        cfg.MODEL.DEVICE = 'cpu'

    return cfg


def train_model(cfg, resume=False):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(cfg.MODEL.WEIGHTS)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()


def load_model(cfg, model_dir=None, model_ckpt="model_final", use_defaultpredictor=True):
    if model_ckpt is None:
        model_ckpt = "model_final"
    if model_dir is None:
        model_dir = cfg.OUTPUT_DIR
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "{}.pth".format(model_ckpt))

    if use_defaultpredictor:
        predictor = DefaultPredictor(cfg)
    else:
        predictor = CustomPredictor(cfg)

    return predictor


def eval_model(cfg, predictor):
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    return inference_on_dataset(predictor.model, val_loader, evaluator)


def save_visualization(vis_dir, db_name='rat_test', predictor=None, save_all=False):
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    rat_metadata = MetadataCatalog.get('rat')
    dataset_dicts = DatasetCatalog.get(db_name)
    if save_all:
        dlist = dataset_dicts
    else:
        dlist = random.sample(dataset_dicts, min(10, len(dataset_dicts)))

    for d in dlist:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=rat_metadata, scale=0.5)
        if predictor is None:
            vis = visualizer.draw_dataset_dict(d)
        else:
            outputs = predictor(img)
            vis = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

        if save_all:
            out_fn = os.path.join(vis_dir, os.path.basename(os.path.dirname(d["file_name"])) + '_' + os.path.basename(
                d["file_name"]))
        else:
            out_fn = os.path.join(vis_dir, os.path.basename(d["file_name"]))
        cv2.imwrite(out_fn, vis.get_image())


class CustomPredictor(DefaultPredictor):
    def __call__(self, original_images):
        with torch.no_grad():
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_images = original_images[:, :, :, ::-1]
            height, width = original_images.shape[1:3]
            inputs = []
            for image in original_images:
                if hasattr(self, 'transform_gen'):
                    image = self.transform_gen.get_transform(image).apply_image(image)
                else:
                    image = self.aug.get_transform(image).apply_image(image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs.append({"image": image, "height": height, "width": width})
            predictions = self.model(inputs)
            return predictions


def read_images_wrapper(params):
    return params[0], utils.io.read_arrays(*params[1:])


def save_result(predictor, seq_dir, out_dir, vis_dir, is_part=False, inst_only=False,
                batch_size=14, visualize=False, n_processes=-1):
    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()

    utils.io.make_directories(out_dir, vis_dir)

    rat_metadata = MetadataCatalog.get('rat')
    fns = utils.io.list_directory(seq_dir)  #, sort_key=utils.io.filename_to_number)

    with multiprocessing.Pool(processes=min(n_processes, batch_size)) as pool:
        batch_start_positions = np.arange(0, len(fns), batch_size, dtype=int)
        batch_end_positions = batch_start_positions + batch_size
        batch_fns = (fns[bstart:bend] for bstart, bend in zip(batch_start_positions, batch_end_positions))
        with tqdm(total=len(batch_start_positions)) as pbar:
            for batch_idx, ims in pool.imap_unordered(read_images_wrapper,
                                                      zip(range(len(batch_start_positions)), batch_fns)):
                all_pred_annot = []
                all_pred_mask = np.zeros(ims.shape[:3], dtype=np.uint8)
                all_pred_vis = np.zeros_like(ims)
                outputs = predictor(ims)
                for frame_id, (im, output) in enumerate(zip(ims, outputs)):
                    segments = output['instances'].to('cpu').pred_masks.numpy()
                    pred_boxes = output['instances'].to('cpu').pred_boxes
                    pred_classes = output['instances'].to('cpu').pred_classes.numpy()
                    scores = output['instances'].to('cpu').scores.numpy()
                    pred_keypoints = None
                    if not is_part and not inst_only:
                        pred_keypoints = output["instances"].to('cpu').pred_keypoints.numpy()

                    pred_annot = dict()
                    pred_mask = np.zeros(im.shape[:2], dtype=np.uint8)
                    for idx in range(len(segments)):
                        pred_mask[segments[idx]] = (idx + 1)
                        pred_annot[idx + 1] = {
                            'pred_boxes': pred_boxes[idx].tensor.to('cpu').numpy().astype('float').tolist(),
                            'pred_classes': int(pred_classes[idx]), 'scores': float(scores[idx]),
                            'pred_masks': utils.segmentation.encode_mask(segments[idx])
                        }
                        if not is_part and not inst_only:
                            pred_annot[idx + 1]['pred_keypoints'] = pred_keypoints[idx].astype('float').tolist()
                    all_pred_annot.append(pred_annot)
                    all_pred_mask[frame_id] = pred_mask

                if visualize:
                    pred_vis = Visualizer(im[:, :, ::-1], metadata=rat_metadata, scale=0.8)
                    pred_vis = pred_vis.draw_instance_predictions(output["instances"].to("cpu"))
                    pred_vis = cv2.resize(pred_vis.get_image(), (im.shape[1], im.shape[0]))
                    all_pred_vis[frame_id] = pred_vis

                utils.io.save_files(out_dir, all_pred_annot, extension='.json', start_idx=batch_idx*batch_size)
                utils.io.save_images(out_dir, all_pred_mask, start_idx=batch_idx*batch_size)
                if visualize:
                    utils.io.save_images(vis_dir, all_pred_vis, start_idx=batch_idx*batch_size)
                pbar.update()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train_annot', help='JSON file containing the annotations of the training set.')
    parser.add_argument('-val', '--val_annot', help='JSON file containing the annotations of the validation set.')
    parser.add_argument('-o', '--out_dir', required=True, help='Output directory.')
    parser.add_argument('--eval_only', default=False, action='store_true', help='Evaluate only.')
    parser.add_argument('-cd', '--model_dir', help='Directory containing the model\'s checkpoint.')
    parser.add_argument('-c', '--model_ckpt', help='Model checkpoint name without extension.', default='model_final')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training.')
    parser.add_argument('--is_part', default=False, action='store_true',
                        help='Part segmentation task, else keypoint detection.')
    parser.add_argument('--inst_only', default=False, action='store_true', help='Do only instance segmentation.')
    parser.add_argument('--eval_dir', help='Evaluate and save results.')
    parser.add_argument('--n_iter', default=10000, type=int, help='Number of training iterations.')
    parser.add_argument('--lr', default=0.00025, type=float, help='Learning rate.')
    parser.add_argument('--visualize', default=False, action='store_true', help='Whether to visualize results or not.')
    parser.add_argument('--cpu_only', default=False, action='store_true', help='Whether only the cpu or not.')
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Whether to overwrite the results of the evaluation or not.')
    return parser.parse_args()


def run(out_dir, train_annot=None, val_annot=None, eval_only=False, model_dir=None, model_ckpt=None, resume=False,
        is_part=False, inst_only=False, eval_dir=None, n_iter=10000, lr=0.00025, visualize=False, cpu_only=False,
        overwrite=False, eval_batch_size=14, n_processes=-1):
    utils.io.remove(os.path.join(out_dir, 'rat_test_coco_format.json'))  # TODO
    # Load datasets
    load_and_register_datasets(train_annot, test_annot=val_annot, is_part=is_part)
    if visualize:
        if val_annot is not None:
            save_visualization(os.path.join(out_dir, 'val'), db_name='rat_test')  # , save_all=True)
        save_visualization(os.path.join(out_dir, 'tr'), db_name='rat_train')


    if model_dir is not None and model_dir.endswith(".pth"):
        model_ckpt = os.path.basename(model_dir[:-4])
        model_dir = os.path.dirname(model_dir)
    # Load configuration
    cfg = set_model_configuration(out_dir, n_iter=n_iter, lr=lr, is_part=is_part, inst_only=inst_only,
                                  model_dir=model_dir, model_ckpt=model_ckpt, cpu_only=cpu_only)

    # Train model
    print("almost training")
    if not eval_only:
        print("start training", cfg.MODEL.WEIGHTS)
        train_model(cfg, resume=resume)

    # Evaluate it on the validation set
    if val_annot is not None:
        predictor = load_model(cfg, model_dir=model_dir, model_ckpt=model_ckpt if not eval_only else None)

        results = eval_model(cfg, predictor)
        out_fn = os.path.join(out_dir, 'val_results')
        out_fn += '_bodypart' if is_part else ('_keypoint' if not inst_only else '_instance')
        out_fn += '.json'
        utils.io.save(out_fn, results)

    # Run model on the given directory
    if eval_dir is not None:
        predictor = load_model(cfg, model_dir=model_dir, model_ckpt=model_ckpt if not eval_only else None,
                               use_defaultpredictor=False)

        seq_dirs = [os.path.join(eval_dir, seq) for seq in os.listdir(eval_dir) if
                    os.path.isdir(os.path.join(eval_dir, seq))]
        if len(seq_dirs) == 0 and 0 < len(os.listdir(eval_dir)):  # Do the prediction on the eval_dir
            seq_dirs = [eval_dir]
        for seq_dir in seq_dirs:
            print(os.path.basename(seq_dir))
            if overwrite or not os.path.exists(os.path.join(out_dir, 'eval_results',
                                                            os.path.basename(seq_dir), 'annot')):
                save_result(predictor, seq_dir,
                            os.path.join(out_dir, 'eval_results', os.path.basename(seq_dir), 'annot'),
                            os.path.join(out_dir, 'eval_results', os.path.basename(seq_dir), 'im'),
                            is_part=is_part, inst_only=inst_only,
                            batch_size=eval_batch_size, visualize=visualize, n_processes=n_processes)
            else:
                print('SKIPPED')


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
