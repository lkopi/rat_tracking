import argparse
import math
import os
import random

import cv2
import numpy as np
import tqdm
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from tensorflow.keras.utils import Sequence

import utils.io
import utils.segmentation
import utils.visualization


def custom_cce_loss(y_true, y_pred):
    # print("y_true", tf.keras.backend.print_tensor(y_true))
    cce = tf.keras.losses.CategoricalCrossentropy()

    bg_cce = cce(tf.keras.backend.reshape(y_true[..., 0], (-1, 1)), tf.keras.backend.reshape(y_pred[..., 0], (-1, 1)))

    tmp1 = cce(tf.keras.backend.reshape(y_true[..., 1:], (-1, 2)), tf.keras.backend.reshape(y_pred[..., 1:], (-1, 2)))
    tmp2 = cce(tf.keras.backend.reshape(y_true[..., 1:][..., ::-1], (-1, 2)), tf.keras.backend.reshape(y_pred[..., 1:], (-1, 2)))
    inst_cce = tf.keras.backend.minimum(tmp1, tmp2)
    return bg_cce + inst_cce


def build_model(shape=(224,224), n_input_channels=8, n_out_channels=3,
                n_layers=3, layer_sizes=[32, 64, 128, 256, 512], use_bottleneck=True):
    """Build a UNet-like segmentation network."""

    def add_activation(curr_out, activation='relu', use_bn=True):
        out = curr_out
        if use_bn:
            out = BatchNormalization()(out)
        out = Activation('relu')(out)
        return out

    def add_block(curr_out, layer_size, kernel_size=3):
        out = Conv2D(layer_sizes[d], 3, padding='same', kernel_initializer='he_normal')(curr_out)
        return add_activation(out)

    assert 4 < n_input_channels

    # Load backbone and get output layers
    backbone = tf.keras.applications.MobileNetV2(include_top=False,
                                                 weights='imagenet',
                                                 input_shape=(shape[0], shape[1], 3))
    bb_outs = [
        backbone.get_layer('block_1_expand_relu').output,
        backbone.get_layer('block_3_expand_relu').output,
        backbone.get_layer('block_6_expand_relu').output,
        backbone.get_layer('block_13_expand_relu').output
    ]

    # Decoder
    n_blocks = len(layer_sizes)
    bb_outs = bb_outs[:n_blocks - 1]
    block_outs = []
    pred_input = Input(shape=(shape[0], shape[1], n_input_channels - 3))
    out = pred_input
    for d in range(n_blocks):
        # Add layers
        for l in range(n_layers):
            out = add_block(out, layer_sizes[d])

        if d < len(bb_outs):
            # Downsample and concatenate backbone
            out = MaxPooling2D(pool_size=(2, 2))(out)
            bout = bb_outs[d]
            if use_bottleneck:
                bout = add_block(bout, layer_sizes[d], kernel_size=1)
            out = Concatenate()([bout, out])
        block_outs.append(out)

    # Encoder
    block_outs = block_outs[::-1]
    layer_sizes = layer_sizes[::-1]
    out = block_outs[0]
    for d in range(1, n_blocks):
        # Upsample
        out = Conv2DTranspose(layer_sizes[d], 2, 2, padding='same', kernel_initializer='he_normal')(out)
        out = add_activation(out)
        out = Concatenate()([block_outs[d], out])

        # Add layers
        for l in range(n_layers):
            out = add_block(out, layer_sizes[d])

    # Add output layer
    out = Conv2D(n_out_channels, 3, padding='same', activation='softmax')(out)

    # Compile the model
    model = Model(inputs=[backbone.input, pred_input], outputs=[out])
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0001))
    model.summary()
    return model


def get_seqs(exp_name, *base_dirs):
    """Get common sequences."""

    if exp_name is None:
        fn_constraint = lambda fn: 'aug' not in fn
    else:
        fn_constraint = lambda fn: exp_name in fn

    seqs = set()
    for bdir in base_dirs:
        seqs &= set(utils.io.list_directory(bdir, full_path=False, only_dirs=True, fn_constraint=fn_constraint))
    return list(seqs)


def load_img(fn, shape=(384,640)):
    """Load, crop, then resize the given image."""

    img = utils.io.read(fn)
    if shape[0] == shape[1]:
        img = utils.segmentation.crop_center(img, cropped_height=420, cropped_width=420)
    img = cv2.resize(img, (shape[1], shape[0]))
    return img


def convert_gt(gt):
    """Convert a label map to a multi-channel image."""

    h, w = gt.shape
    ids = np.unique(gt)
    out = np.zeros((h, w, len(ids)), dtype=np.float32)
    for i, idx in enumerate(ids):
        out[gt == idx, i] = 1.
    return out


def load_sample(seq, img_dir_base, pred_dir_base, gt_dir_base=None,
                shape=(384,640), frame_idx=None,
                reuse_outputs=False, max_dist=15):
    """Load a random sample from the given sequence."""

    img_dir = os.path.join(img_dir_base, seq)
    if gt_dir_base is not None: gt_dir = os.path.join(gt_dir_base, seq)
    pred_dir = os.path.join(pred_dir_base, seq)

    # Get a random filename
    fns = utils.io.list_directory(img_dir, sort_key=utils.io.filename_to_number, full_path=False)
    curr_frame = np.random.randint(len(fns)) if frame_idx is None else frame_idx

    # Load sample
    base_fn, ext = os.path.splitext(fns[curr_frame])
    curr_img = load_img(os.path.join(img_dir, fns[curr_frame]), shape=shape) / 127.5 - 1
    if gt_dir_base is not None:
        # curr_gt = convert_gt(load_img(os.path.join(gt_dir, fns[curr_frame]), shape=shape))
        curr_gt = convert_gt(load_img(os.path.join(gt_dir, '{}_instances{}'.format(base_fn, ext)), shape=shape))
    else:
        curr_gt = None

    curr_insts = load_img(os.path.join(pred_dir, '{}_instances{}'.format(base_fn, ext)), shape=shape).astype(np.float32)
    curr_insts /= max(1, np.max(curr_insts))
    curr_parts = load_img(os.path.join(pred_dir, '{}_parts{}'.format(base_fn, ext)), shape=shape) / 3
    curr_kps = load_img(os.path.join(pred_dir, '{}_keypoints{}'.format(base_fn, ext)), shape=shape) / 255

    curr_preds = np.dstack((curr_insts[..., None], curr_parts[..., None], curr_kps))

    if reuse_outputs:
        # Load a random gt and concat it to the input
        if (0 < curr_frame) and (gt_dir_base is not None):
            if frame_idx is not None:
                dist = 1
            else:
                dist = min(curr_frame, max_dist)
                dist = np.random.randint(dist)
            if dist == 0:
                prev_gt = np.zeros_like(curr_img)
            else:
                prev_frame = curr_frame - dist
                prev_gt = convert_gt(load_img(os.path.join(gt_dir,
                                                           '{}_instances{}'.format(os.path.splitext(fns[prev_frame])[0],
                                                                                                    ext)), shape=shape))
        else:
            prev_gt = np.zeros_like(curr_img)
        curr_preds = np.dstack((curr_preds, prev_gt)), curr_gt

    return (curr_img, curr_preds), curr_gt


class RatSequence(Sequence):
    """Datagenerator."""

    def __init__(self, exp_names, img_dir_base, gt_dir_base, pred_dir_base, batch_size, shape=(224,224),
                 reuse_outputs=False, max_dist=15):
        self.exp_names = exp_names
        self.img_dir_base = img_dir_base
        self.gt_dir_base = gt_dir_base
        self.pred_dir_base = pred_dir_base
        self.batch_size = batch_size
        self.shape = shape
        self.reuse_outputs = reuse_outputs
        self.max_dist = max_dist
        # Get sequences
        self.seqs = []
        for exp_name in self.exp_names:
            self.seqs += get_seqs(exp_name, self.img_dir_base, self.gt_dir_base, self.pred_dir_base)

    def __len__(self):
        return math.ceil(len(self.seqs) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(len(self.seqs), (idx + 1) * self.batch_size)
        batch_len = end_idx - start_idx

        # Initialize
        pred_len = 8 if self.reuse_outputs else 5
        x_imgs = np.empty((batch_len, self.shape[0], self.shape[1], 3), dtype=np.float32)
        x_pred = np.empty((batch_len, self.shape[0], self.shape[1], pred_len), dtype=np.float32)
        y_out = np.empty((batch_len, self.shape[0], self.shape[1], 3), dtype=np.int32)

        # Load samples
        for i, idx in enumerate(range(start_idx, end_idx)):
            (img, pred), gt = load_sample(self.seqs[idx], self.img_dir_base, self.pred_dir_base,
                                          gt_dir_base=self.gt_dir_base, shape=self.shape,
                                          reuse_outputs=self.reuse_outputs, max_dist=self.max_dist)
            x_imgs[i] = img
            x_pred[i] = pred
            y_out[i] = gt

        return (x_imgs, x_pred), y_out

    def on_epoch_end(self):
        random.shuffle(self.seqs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', required=True, help='Experiment name.')
    parser.add_argument('-i', '--img_dir_base', required=True, help='Base directory of the RGB images.')
    parser.add_argument('-p', '--pred_dir_base', required=True, help='Base directory of the predictions.')
    parser.add_argument('-o', '--out_dir_base', required=True, help='Base directory of the predictions.')
    parser.add_argument('-g', '--gt_dir_base', help='Base directory of the GT masks.')
    parser.add_argument('-l', '--load_weights', help='Weights of a pretrained network.')
    parser.add_argument('--eval_only', default=False, action='store_true', help='Evaluate only.')
    parser.add_argument('--only_original', default=False, action='store_true', help='Train only on original samples.')
    parser.add_argument('--save_predictions', default=False, action='store_true', help='Train only on original samples.')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size.')
    return parser.parse_args()


def run(exp_name, img_dir_base, pred_dir_base, out_dir_base, gt_dir_base=None, eval_only=False, only_original=False,
        save_predictions=False, batch_size=4):
    checkpoint_filepath = os.path.join(out_dir_base, exp_name)
    exp_names = [None, 'default', 'scalerot_aug', 'scalerotwarp']
    if only_original:
        exp_names = [None]

    if not eval_only:
        train_generator = RatSequence(exp_names, img_dir_base, gt_dir_base, pred_dir_base, batch_size)

        val_img_dir_base = "/media/hdd2/lkopi/datasets/rats/temporal_extension/val_video_masks_both/img"
        val_gt_dir_base = "/media/hdd2/lkopi/datasets/rats/temporal_extension/val_video_masks_both/annot"
        val_pred_dir_base = "/media/hdd2/lkopi/datasets/rats/temporal_extension/val_video_masks_both/tracktor_annot"
        val_generator = RatSequence(exp_names, val_img_dir_base, val_gt_dir_base, val_pred_dir_base, batch_size)

    # Load model
    model = build_model(shape=(224, 224))
    if not eval_only:
        # Train model
        if load_weights:
            model.load_weights(load_weights)
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        model_chkpnt = ModelCheckpoint(filepath=checkpoint_filepath,
                                       save_weights_only=True)
        tensorboard = TensorBoard(log_dir=checkpoint_filepath)
        history = model.fit(train_generator,
                            validation_data=val_generator,
                            verbose=1,
                            epochs=30,  # 30,
                            callbacks=[model_chkpnt, tensorboard])  # , early_stopping])
    model.load_weights(checkpoint_filepath)

    # Evaluate
    if save_predictions:
        seqs = get_seqs(None, img_dir_base, pred_dir_base)

        for seq_name in seqs:
            print(seq_name)
            fns = utils.io.list_directory(os.path.join(img_dir_base, seq_name), sort_key=utils.io.filename_to_number, full_path=False)
            prev_out = None
            imgs = np.empty((len(fns), 224, 224, 3), dtype=np.uint8)
            outs = np.empty((len(fns), 224, 224, 3), dtype=np.uint8)
            for i, fn in tqdm.tqdm(enumerate(fns)):
                (img, pred), gt = load_sample(seq_name, img_dir_base, pred_dir_base,
                                              gt_dir_base=pred_dir_base, shape=(224, 224), frame_idx=i)
                # if 0 < i:
                #    pred[..., -3:] = prev_out
                prev_out = model.predict((img[None], pred[None]))[0]
                imgs[i] = ((img+1)*127.5).astype(np.uint8)
                outs[i] = (np.round(prev_out)*255).astype(np.uint8)
            utils.io.save_video(os.path.join(checkpoint_filepath, 'outs', seq_name + '.avi'),
                                utils.visualization.concat_images([imgs, outs]))


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
