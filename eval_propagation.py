#!/usr/bin/env python
# coding: utf-8

import numpy as np

import utils.io
import utils.segmentation
import utils.visualization


def count_track_switches(gts, preds, return_switched_frames=False):
    # Invert instance ids
    tmp = np.zeros_like(preds)
    fst_id, snd_id = np.unique(preds)[1:]  # idtracker.ai: remove
    # fst_id, snd_id = np.unique(gts)[1:]  # idtracker.ai: add
    tmp[preds == fst_id] = snd_id
    tmp[preds == snd_id] = fst_id

    fixed_preds = preds.copy()
    n_switches = 0
    switched_frames = []
    for i in range(len(gts)):
        if np.count_nonzero(gts[i] == preds[i]) < np.count_nonzero(gts[i] == tmp[i]):
            preds, tmp = tmp, preds
            fixed_preds[i:] = preds[i:].copy()
            if i != 0:
                n_switches += 1
                switched_frames.append(i)
    if return_switched_frames:
        return n_switches, fixed_preds, switched_frames
    return n_switches, fixed_preds


def calculate_iou(gtmask, labelmask):
    intersect = np.count_nonzero(gtmask * labelmask)
    union = np.count_nonzero(labelmask + gtmask)
    return intersect / union


def evaluate_sequence(seq, gt_dir, pred_dir, return_seq=True):
    gts, preds, n_frames = load_sequence(gt_dir, seq, pred_dir)

    n_track_switches, avg_iou = evaluate_prediction(gts, preds)

    if return_seq:
        return n_track_switches, avg_iou, n_frames, seq
    return n_track_switches, avg_iou, n_frames


def calulate_average_iou(gts, preds):
    # assert np.all(np.unique(gts) == np.unique(preds)):
    unique_ids = list(set(list(np.unique(gts)[1:]) + list(np.unique(preds)[1:])))
    if len(unique_ids) == 2:
        fst_id, snd_id = unique_ids

        # fst_id, snd_id = np.unique(preds)[1:]
        iou1 = calculate_iou(gts == fst_id, preds == fst_id)
        iou2 = calculate_iou(gts == snd_id, preds == snd_id)
        avg_iou1 = (iou1 + iou2) / 2

        iou1 = calculate_iou(gts == fst_id, preds == snd_id)
        iou2 = calculate_iou(gts == snd_id, preds == fst_id)
        avg_iou2 = (iou1 + iou2) / 2
        iou = max(avg_iou1, avg_iou2)
    else:
        fst_id, = unique_ids
        iou = calculate_iou(gts == fst_id, preds == fst_id) / 2
    return iou


def evaluate_prediction(gts, preds):
    gt_ids, pred_ids = np.unique(gts), np.unique(preds)
    assert len(gt_ids) == len(pred_ids), "{}, {}".format(gt_ids, pred_ids)  # idtracker.ai: remove
    if not np.all(gt_ids == pred_ids):
        preds = utils.visualization.map_labels(preds, dict(zip(pred_ids, gt_ids)))[..., 0]

    n_track_switches, fixed_preds = count_track_switches(gts, preds)

    #if len(pred_ids) <= 2:   # idtracker.ai: add
    #    fst_id = pred_ids[1]
    #    avg_iou = calculate_iou(gts == fst_id, fixed_preds == fst_id)
    #else:
    fst_id, snd_id = np.unique(fixed_preds)[1:]
    iou1 = calculate_iou(gts == fst_id, fixed_preds == fst_id)
    iou2 = calculate_iou(gts == snd_id, fixed_preds == snd_id)
    avg_iou = (iou1 + iou2) / 2

    return n_track_switches, avg_iou


def load_sequence(gt_dir, seq, pred_dir):
    # Load GT
    gts = utils.io.read_arrays(utils.io.list_directory('{}/{}'.format(gt_dir, seq),
                                                       sort_key=utils.io.filename_to_number,
                                                       extension='.png'))
    if np.any(gts >= 127):
        gts //= 127

    # Load predictions
    # preds = utils.read_frames_from_video('{}/{}.avi'.format(pred_dir, seq))[0]
    # preds = utils.io.read_arrays(utils.io.list_directory('{}/{}_instances'.format(pred_dir, seq),
    # print(utils.io.list_directory('{}/{}'.format(pred_dir, seq), sort_key=utils.io.filename_to_number, extension='.png'))
    preds = utils.io.read_arrays(utils.io.list_directory('{}/{}'.format(pred_dir, seq),
                                                         sort_key=utils.io.filename_to_number,
                                                         extension='.png'))
    preds = utils.segmentation.multi2singlechannel(preds)  # idtracker.ai: remove
    # print(f"{seq}: {preds.ndim}, {preds.shape}")
    # preds = np.round(preds / 127).astype(np.uint8)  # TODO: remove (Anna)
    n_frames, h, w = gts.shape
    # preds = preds[:, h:, w:]
    tmp = np.zeros_like(gts)
    if preds.ndim == 3:
        inst_ids = np.unique(preds)[1:]
        tmp[preds == inst_ids[0]] = 1
        if 1 < len(inst_ids):
            tmp[preds == inst_ids[1]] = 2
    elif preds.ndim == 4:
        inst_ids = []
        for ch in range(preds.shape[-1]):
            if 0 < np.count_nonzero(preds[..., ch] > 127):
                inst_ids.append(ch)
        tmp[preds[..., inst_ids[0]] > 127] = 1
        tmp[preds[..., inst_ids[1]] > 127] = 2
    else:
        raise NotImplementedError(f"{seq}: {preds.ndim}, {preds.shape}")
    assert len(inst_ids) == 2, "{}, {}".format(inst_ids, seq)  # idtracker.ai: remove
    preds = tmp
    assert gts.shape == preds.shape, "{}, {}, {}".format(seq, gts.shape, preds.shape)  # idtracker.ai: remove
    return gts, preds, n_frames
