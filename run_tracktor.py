#!/usr/bin/env python
# coding: utf-8

import sys

import utils.io
import utils.visualization

sys.path.append('/home/lkopi/repo/tracking_wo_bnw/src')
sys.path.append('/home/lkopi/repo/phd_utils/src')

import argparse
import os
import time

# import some common libraries
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import motmetrics as mm
mm.lap.default_solver = 'lap'
import torch
from torch.utils.data import DataLoader
import torchvision
import tqdm

# import some detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# import some tracktor utilities
from tracktor.detectron2prop import Detectron2Prop
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.detectron2tracker import Detectron2Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums

import train_detectron2 as td2

from phdutils import io, visualization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', required=True, help='Input directory containing rgb images.')
    parser.add_argument('-o', '--out_dir', required=True, help='Output directory.')
    parser.add_argument('-d', '--detectron_ckpt', help='Path to a detectron2 model checkpoint.', default='/media/hdd2/lkopi/datasets/rats/augmentation_test/detectron2_out/00019_keypoint/model_0029999.pth')
    parser.add_argument('-r', '--reid_ckpt', help='Path to a reid model checkpoint.', default='/home/lkopi/repo/tracking_wo_bnw/output/tracktor/reid/res50-mot17-batch_hard/ResNet_iter_25245.pth')
    parser.add_argument('--is_part', default=False, action='store_true', help='Part segmentation task, else keypoint detection.')
    parser.add_argument('--inst_only', default=False, action='store_true', help='Do only instance segmentation.')
    parser.add_argument('--do_reid', default=False, action='store_true', help='Apply the reidentication network.')
    parser.add_argument('--visualize', default=False, action='store_true', help='Whether to visualize results or not.')
    return parser.parse_args()


def load_obj_det_model(detectron_ckpt, is_part=False, inst_only=False):
    assert os.path.isfile(detectron_ckpt) and detectron_ckpt[-3:] == 'pth'

    cfg = td2.set_model_configuration(os.path.dirname(detectron_ckpt), is_part=is_part, inst_only=inst_only)
    cfg.MODEL.WEIGHTS = detectron_ckpt
    predictor = DefaultPredictor(cfg)

    return Detectron2Prop(predictor.model)


def load_reid_network(reid_ckpt):
    assert os.path.isfile(reid_ckpt) and reid_ckpt[-3:] == 'pth'

    reid_network = resnet50(pretrained=False, output_dim=128)
    reid_network.load_state_dict(torch.load(reid_ckpt,
                                            map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()
    
    return reid_network


def load_tracker(detectron_ckpt, reid_ckpt, is_part=False, inst_only=False, do_reid=False):
    '''
    tracker_conf = {
        # FRCNN score threshold for detections
        'detection_person_thresh': 0.5,
        # FRCNN score threshold for keeping the track alive
        'regression_person_thresh': 0.5,
        # NMS threshold for detection
        'detection_nms_thresh': 0.6, # 1.0, # 0.3,
        # NMS theshold while tracking
        'regression_nms_thresh': 0.8, # 1.0, # 0.6,
        # motion model settings
        'motion_model': {
        'enabled': False,
        # average velocity over last n_steps steps
        'n_steps': 1,
        # if true, only model the movement of the bounding box center. If false, width and height are also modeled.
        'center_only': True
        },
        # DPM or DPM_RAW or 0, raw includes the unfiltered (no nms) versions of the provided detections,
        # 0 tells the tracker to use private detections (Faster R-CNN)
        'public_detections': False,  # <-- True
        # How much last appearance features are to keep
        'max_features_num': 100,
        # Do camera motion compensation
        'do_align': False,  # <-- True
        # Which warp mode to use (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, ...)
        'warp_mode': str(1), # cv2.MOTION_EUCLIDEAN,
        # maximal number of iterations (original 50)
        'number_of_iterations': 100,
        # Threshold increment between two iterations (original 0.001)
        'termination_eps': 0.00001,
        # Use siamese network to do reid
        'do_reid': do_reid,  # <-- True
        # How much timesteps dead tracks are kept and cosidered for reid
        'inactive_patience': 100,
        # How similar do image and old track need to be to be considered the same person
        'reid_sim_threshold': 1.0, # 2.0,
        # How much IoU do track and image need to be considered for matching
        'reid_iou_threshold': 0.2
    }
    '''
    tracker_conf = {
        # FRCNN score threshold for detections
        'detection_person_thresh': 0.5,
        # FRCNN score threshold for keeping the track alive
        'regression_person_thresh': 0.5,
        # NMS threshold for detection
        'detection_nms_thresh': 0.3,
        # NMS theshold while tracking
        'regression_nms_thresh': 0.6,
        # motion model settings
        'motion_model': {
        'enabled': False,
        # average velocity over last n_steps steps
        'n_steps': 1,
        # if true, only model the movement of the bounding box center. If false, width and height are also modeled.
        'center_only': True
        },
        # DPM or DPM_RAW or 0, raw includes the unfiltered (no nms) versions of the provided detections,
        # 0 tells the tracker to use private detections (Faster R-CNN)
        'public_detections': False,  # <-- True
        # How much last appearance features are to keep
        'max_features_num': 100,
        # Do camera motion compensation
        'do_align': False,  # <-- True
        # Which warp mode to use (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, ...)
        'warp_mode': str(1), # cv2.MOTION_EUCLIDEAN,
        # maximal number of iterations (original 50)
        'number_of_iterations': 100,
        # Threshold increment between two iterations (original 0.001)
        'termination_eps': 0.00001,
        # Use siamese network to do reid
        'do_reid': do_reid,  # <-- True
        # How much timesteps dead tracks are kept and cosidered for reid
        'inactive_patience': 100,
        # How similar do image and old track need to be to be considered the same person
        'reid_sim_threshold': 2.0,
        # How much IoU do track and image need to be considered for matching
        'reid_iou_threshold': 0.2
    }
    obj_detect = load_obj_det_model(detectron_ckpt, is_part=is_part, inst_only=inst_only)
    reid_network = load_reid_network(reid_ckpt)
    
    tracker = Detectron2Tracker(obj_detect, reid_network, tracker_conf)
    tracker.reset()

    return tracker

    
def load_images(in_dir):
    fns = utils.io.list_directory(in_dir, sort_key=utils.io.filename_to_number)
    return utils.io.read_arrays(fns, utils.io.read)


def prepare_output(results, n_frames):
    out = {}
    for i in range(n_frames):
        out[i] = {}
        for k in results:
            if i in results[k]:
                res = results[k][i][0]
                pclass, mask, kps = results[k][i][1:]
                
                bbox = np.rint(res[:4].astype(np.float32)).astype(np.int32).tolist()
                score = float(res[-1].detach().to('cpu').numpy())
                
                x1, y1, x2, y2 = bbox
                mask = mask[y1:y2, x1:x2].tolist()
                if kps is not None:
                    kps = np.rint(kps.astype(np.float32)).astype(np.int32).tolist()
                    out[i][k] = (bbox, score, int(pclass), mask, kps)
                else:
                    out[i][k] = (bbox, score, int(pclass), mask)
    return out


def save_track_matrix(basefn, results, n_frames, visualize=True):
    n_inst = len(results)
    out = np.zeros((n_inst, n_frames, 6), dtype=np.float32)
    for i in range(n_frames):
        for k in results:
            if i in results[k]:
                res = results[k][i][0]
                pclass = results[k][i][1]
                bbox = np.rint(res[:4].astype(np.float32))
                score = res[-1].detach().to('cpu').numpy()
                
                out[k, i, :4] = bbox
                out[k, i,  4] = pclass
                out[k, i, -1] = score
    utils.io.save(basefn + '.npy', out)

    # Visualize tracks
    plt.figure(figsize=(100, 5))
    plt.imshow((out[..., -1] > 0) * (out[..., -2] + 1))
    plt.xticks(ticks=np.arange(n_frames+1)-.5, labels=np.arange(n_frames)+1)
    plt.yticks(ticks=np.arange(n_inst+1)-.5, labels=np.arange(n_inst)+1)
    plt.grid()
    plt.colorbar(ticks=range(int(out[..., -2].max())+2))
    plt.savefig(basefn + '.png')
    plt.clf()

def color_tracks(seqs, results, n_frames):
    np.random.seed(0)
    colors = utils.visualization.generate_random_colors(1000)

    outs = seqs.copy()
    n_inst = len(results)
    for i in range(n_frames):
        for k in results:
            if i in results[k]:
                x1, y1, x2, y2, _ = results[k][i][0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                outs[i] = cv2.rectangle(outs[i], (x1, y1), (x2, y2), colors[k], 2)

                mask = results[k][i][-2]
                #tmp = cv2.resize(outs[i], (640, 480))
                #tmp = visualization.overlay_mask(tmp, mask, color=colors[k])
                #outs[i] = cv2.resize(tmp, (640, 420))
                outs[i] = visualization.overlay_mask(outs[i], mask, color=colors[k])

                kps = results[k][i][-1]
                if kps is not None:
                    # tmp = cv2.resize(outs[i], (640, 480))
                    kps = np.rint(kps.astype(np.float32)).astype(np.int32).tolist()
                    for j, kp in enumerate(kps):
                        c = colors[min(k*n_inst+j, len(colors))]
                        x, y, _ = kp 
                        # tmp = cv2.circle(tmp, center=(x, y), radius=2, color=c, thickness=2)
                        outs[i] = cv2.circle(outs[i], center=(x, y), radius=2, color=c, thickness=2)
                    # outs[i] = cv2.resize(tmp, (640, 420))
    return outs


def run(in_dir, out_dir, detectron_ckpt, reid_ckpt, is_part=False, inst_only=False, do_reid=False, visualize=False):
    # Create output dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize tracker
    tracker = load_tracker(detectron_ckpt, reid_ckpt, is_part=is_part, inst_only=inst_only, do_reid=do_reid)
    seqs = load_images(in_dir)
    
    # Run tracker
    n_frames = 0
    for i, frame in tqdm.tqdm(enumerate(seqs)):
        blob = {'img': torch.tensor(np.transpose(frame/255, (2, 0, 1)).astype(np.float32)[None])}
        with torch.no_grad():
            tracker.step(blob)
        n_frames += 1
    results = tracker.get_results()


    # Save results
    out_basefn = os.path.join(out_dir, '{}_{}{}'.format(os.path.basename(in_dir),
                                                        'parts' if is_part else 'kp',
                                                        '_reid' if do_reid else ''))
    out = prepare_output(results, n_frames)
    utils.io.save(out_basefn + '.json', out)

    # Save numpy track matrix
    save_track_matrix(out_basefn, results, n_frames, visualize=visualize)

    # Visualize results
    if visualize:
        outs = color_tracks(seqs, results, n_frames)
        utils.io.save_video(out_basefn + '.avi', outs)


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
