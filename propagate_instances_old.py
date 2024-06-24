#!/usr/bin/env python
# coding: utf-8

import itertools
import multiprocessing
import os

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from skimage.measure import label
from skimage.morphology import convex_hull_image, closing, disk, remove_small_holes
from tqdm import tqdm

import utils.io
import utils.segmentation
import utils.visualization


def single2multichannel(masks):
    l, h, w = masks.shape
    ch = np.unique(masks)[-1]
    out = np.zeros((l, h, w, ch), dtype=masks.dtype)
    for c in range(ch):
        tmp = (masks == c + 1)
        out[tmp, c] = c + 1
    return out


def multi2singlechannel(masks):
    l, h, w, ch = masks.shape
    out = np.zeros((l, h, w), dtype=masks.dtype)
    for c in range(0, ch):
        tmp = (masks[..., c] > 0)
        out[tmp] = masks[tmp, c]
    return out


def read_detectron2_output(out_dir, read_json=False, segments_from_json=False, multichannel=False,
                           return_classes=False):
    '''Read the results of detectron2'''
    assert not (not read_json and segments_from_json)  # segments_from_json can only be true, if read_json is true

    fns = utils.io.list_directory(out_dir, sort_key=utils.io.filename_to_number, extension='.png')

    instances = utils.io.read_arrays(fns)
    if multichannel:
        instances = single2multichannel(instances)

    if read_json:
        jsons = utils.io.list_directory(out_dir, sort_key=utils.io.filename_to_number, extension='.json')
        annots = utils.io.read_files(jsons)

        # Load instances from the json file
        if segments_from_json:
            # Initialize instances
            ch, h, w = instances.shape[:3]
            if not multichannel:
                n_insts = np.unique(instances)[-1]
                instances = np.zeros((ch, h, w, n_insts), dtype=np.uint8)
            else:
                instances *= 0  # Shape is already good, set to 0
                if return_classes:
                    classes = np.copy(instances)

            for frame_id, annot in enumerate(annots):
                for inst_id, anns in annot.items():
                    inst_id = int(inst_id)
                    for segment in anns['pred_masks']:
                        decoded_mask = utils.segmentation.decode_mask(segment, (h, w))
                        instances[frame_id, decoded_mask, inst_id - 1] = inst_id
                        if return_classes:
                            classes[frame_id, decoded_mask, inst_id - 1] = int(anns['pred_classes']) + 1

            if not multichannel:
                instances = multi2singlechannel(instances)
        if not return_classes:
            return instances, annots
        else:
            return instances, annots, classes
    return instances


def remove_small_regions(labels, bg_label=-1, min_area=50):
    """Remove regions that are smaller than `min_area`.

    Parameters
    ----------
    labels : np.ndarray
        2D matrix containing the labels
    bg_label : int, optional
        Background label id, by default -1
    min_area : int, optional
        Minimum allowed area of each region, by default 50

    Returns
    -------
    np.ndarray
        Label matrix where the labels of regions that are smaller than `min_area` was set to `bg_label`.
    """
    relabeled_img = label(labels, background=bg_label)
    relabeled_ids, areas = np.unique(relabeled_img.reshape(-1), return_counts=True)

    regions = zip(relabeled_ids, areas)
    filtered_regions = filter(lambda label_area: label_area[1] < min_area, regions)
    filtered_label_ids = list(map(lambda label_area: label_area[0], filtered_regions))

    mask = np.isin(relabeled_img, filtered_label_ids)

    filtered_labels = np.copy(labels)
    filtered_labels[mask.reshape(labels.shape)] = bg_label
    return filtered_labels


def apply_closing(labels, disk_size=6):
    labels = labels.copy()
    label_ids = np.unique(labels)[1:]
    selem = disk(disk_size)

    for label_id in label_ids:
        mask = (labels == label_id)
        mask = closing(mask, selem)
        labels[mask] = label_id
    return labels


def fill_small_holes(labels, area_threshold=50):
    """Fill regions that are smaller than `area_threshold`.

    Parameters
    ----------
    labels : np.ndarray
        2D matrix containing the labels
    area_threshold : int, optional
        Minimum allowed area of each region, by default 50

    Returns
    -------
    np.ndarray
        Label matrix where no regions with smaller than `area_threshold` holes are present.
    """
    labels = labels.copy()
    label_ids = np.unique(labels)[1:]

    for label_id in label_ids:
        mask = (labels == label_id)
        mask = remove_small_holes(mask, area_threshold=area_threshold)
        labels[mask] = label_id
    return labels


def get_inst_ids_and_area(frame):
    inst_ids, counts = np.unique(frame, return_counts=True)
    if inst_ids[0] == 0:
        inst_ids, counts = inst_ids[1:], counts[1:]
    return inst_ids, counts


def clean_frame(idx, frame, min_inst_per_frame=None, max_inst_per_frame=None, remove_incosistencies=True,
                remove_small_region=True, min_area=200, closing_area=10, overlap_thrs=0.1, check_shape=True,
                remove_thorned_regions=False, thorned_thrs=1., multichannel=False, drop_unused_channels=True):
    # Smooth instances
    if remove_small_region:
        if not multichannel:
            frame = frame[..., None]
        for inst_id in range(frame.shape[-1]):
            frame[..., inst_id] = remove_small_regions(frame[..., inst_id], bg_label=0, min_area=min_area)
            if 0 < closing_area and 0 < np.count_nonzero(frame[..., inst_id]):
                frame[..., inst_id] = apply_closing(frame[..., inst_id], disk_size=closing_area)
            frame[..., inst_id] = fill_small_holes(frame[..., inst_id], area_threshold=min_area)
        if not multichannel:
            frame = frame[..., 0]

    inst_ids, counts = get_inst_ids_and_area(frame)
    if min_inst_per_frame is not None and len(inst_ids) < min_inst_per_frame:
        frame *= 0
        return idx, frame

    if check_shape and (min(counts) / max(counts) < 0.4):  # only for 2 instances of the same class
        frame[frame == inst_ids[np.argmin(counts)]] = 0
        # Recalculate instance indices and sizes
        inst_ids, counts = get_inst_ids_and_area(frame)

    if remove_thorned_regions:
        for inst_id in inst_ids:
            inst_mask = (frame == inst_id)
            if multichannel:
                inst_mask = inst_mask[..., inst_id - 1]

            if thorned_thrs == 1.:
                _, n_labels = label(inst_mask, connectivity=2, return_num=True)
                if 1 < n_labels:
                    if not multichannel:
                        frame[inst_mask] = 0
                    else:
                        frame[inst_mask, inst_id - 1] = 0
            elif 0. < thorned_thrs:
                chull = convex_hull_image(inst_mask)
                if (np.count_nonzero(inst_mask) / np.count_nonzero(chull)) < thorned_thrs:
                    if not multichannel:
                        frame[inst_mask] = 0
                    else:
                        frame[inst_mask, inst_id - 1] = 0
        # Recalculate instance indices and sizes
        inst_ids, counts = get_inst_ids_and_area(frame)

    if remove_incosistencies:
        for inst_id in inst_ids:
            inst_mask = (frame == inst_id)
            if 0 == np.count_nonzero(inst_mask):
                continue
            if multichannel:
                inst_mask = inst_mask[..., inst_id - 1]
            # inst_mask = remove_small_objects(inst_mask, min_size=256)
            chull = convex_hull_image(inst_mask)
            if multichannel:
                chull = np.stack([chull] * frame.shape[-1], axis=-1)

            tmp = frame.copy()
            if not multichannel:
                tmp[inst_mask] = 0
            else:
                tmp[inst_mask, inst_id - 1] = 0

            ious, label_ids = utils.segmentation.mask_intersection_over_union(tmp, chull)
            label_mask = (ious > overlap_thrs)
            ious, label_ids = ious[label_mask], label_ids[label_mask]
            if len(label_ids) != 0:
                if not multichannel:
                    frame[np.isin(tmp, label_ids)] = 0
                else:
                    for lid in label_ids:
                        frame[..., lid - 1] = 0
        # Recalculate instance indices and sizes
        inst_ids, counts = get_inst_ids_and_area(frame)

    if (max_inst_per_frame is not None and len(inst_ids) > max_inst_per_frame):
        frame *= 0
    # if min_inst_per_frame is not None and len(inst_ids) < min_inst_per_frame:
    #     frame *= 0

    return idx, frame


def clean_instances(instances, min_inst_per_frame=None, max_inst_per_frame=None, remove_incosistencies=True,
                    remove_small_regions=True, min_area=200, closing_area=10, overlap_thrs=0.1, check_shape=True,
                    remove_thorned_regions=False, thorned_thrs=1., multichannel=False, overwrite=False):
    if overwrite:
        instances = instances.copy()
    for idx, frame in tqdm(enumerate(instances)):
        i, f = clean_frame(idx, frame, min_inst_per_frame, max_inst_per_frame, remove_incosistencies,
                           remove_small_regions, min_area, closing_area, overlap_thrs, check_shape,
                           remove_thorned_regions, thorned_thrs, multichannel)
        instances[i] = f
    return instances


def clean_frame_wrapper(params):
    return clean_frame(*params)


def clean_instances_parallel(instances, min_inst_per_frame=None, max_inst_per_frame=None, remove_incosistencies=True,
                             remove_small_regions=True, min_area=200, closing_area=10, overlap_thrs=0.1,
                             check_shape=True,
                             remove_thorned_regions=False, thorned_thrs=1., multichannel=False):
    instances = instances.copy()
    with multiprocessing.Pool() as pool:
        with tqdm(total=len(instances)) as pbar:
            for idx, frame in pool.imap_unordered(clean_frame_wrapper,
                                                  zip(range(len(instances)),
                                                      instances,
                                                      itertools.repeat(min_inst_per_frame),
                                                      itertools.repeat(max_inst_per_frame),
                                                      itertools.repeat(remove_incosistencies),
                                                      itertools.repeat(remove_small_regions),
                                                      itertools.repeat(min_area),
                                                      itertools.repeat(closing_area),
                                                      itertools.repeat(overlap_thrs),
                                                      itertools.repeat(check_shape),
                                                      itertools.repeat(remove_thorned_regions),
                                                      itertools.repeat(thorned_thrs),
                                                      itertools.repeat(multichannel))):
                instances[idx] = frame
                pbar.update()
    return instances


def mask_intersection_over_union(maskA, maskB, use_iou=False):
    label_ids, intersect_counts = np.unique(maskA[maskB > 0], return_counts=True)
    _, all_counts = np.unique(maskA[np.isin(maskA, label_ids)], return_counts=True)
    ious = intersect_counts / (np.count_nonzero(maskB) if use_iou else all_counts)
    if len(label_ids) == 0:
        return 0.
    elif label_ids[0] == 0:
        if 1 < len(label_ids):
            return ious[1]
        else:
            return 0.
    else:
        return ious[0]


def match_proposals(curr_masks, next_masks):
    curr_ids = np.unique(curr_masks)[1:]
    next_ids = np.unique(next_masks)[1:]
    IOU_mat = np.zeros((len(curr_ids), len(next_ids)), dtype=np.float32)
    for t, tid in enumerate(curr_ids):
        for d, did in enumerate(next_ids):
            IOU_mat[t, d] += mask_intersection_over_union(curr_masks == tid, next_masks == did)
    matched_indices = linear_assignment(-IOU_mat)
    return matched_indices, IOU_mat, curr_ids, next_ids


def check_consistency(preds, iou_thres=0.8, overwrite=False):
    ''' Check the segmentation (and bbox) consistency between 2 subsequent frames.
    Return False, if
        the iou < `iou_thres`
        the bbox_iou < `bbox_thres`
    Otherwise relabel the predictions based on the prev frames prediction
    '''
    if overwrite:
        preds = preds.copy()
    init_idx = 0
    while init_idx < len(preds) and len(np.unique(preds[init_idx])) < 3:
        init_idx += 1
    if init_idx == len(preds):
        return preds, list(range(len(preds)))

    prev_pred = preds[init_idx].copy()
    bad_frame_ids = []
    for idx, pred in enumerate(preds):
        if init_idx == idx:
            continue
        matched_indices, IOU_mat, curr_ids, next_ids = match_proposals(prev_pred, pred)
        # print(matched_indices, IOU_mat, curr_ids, next_ids)
        curr_pred = pred.copy()
        track_map = dict(zip(next_ids[matched_indices[1]].tolist(), curr_ids[matched_indices[0]].tolist()))
        for k in track_map.keys():
            preds[idx, curr_pred == k] = track_map[k]
        if len(track_map) < 2:
            keep_map = dict(zip(list(set(curr_ids) - set(track_map.keys())),
                                list(set(curr_ids) - set(track_map.values()))))
            for k in keep_map.keys():
                preds[idx, curr_pred == k] = 0
                preds[idx, prev_pred == keep_map[k]] = keep_map[k]
            bad_frame_ids.append(idx)
        prev_pred = preds[idx]
    return preds, bad_frame_ids


def propagate_instances(instances, overlap_thrs=0.1):
    cleaned = clean_instances_parallel(instances, min_inst_per_frame=2, max_inst_per_frame=2,
                                       closing_area=5, overlap_thrs=overlap_thrs)
    cons, bad_frame_ids = check_consistency(cleaned)
    return cons, bad_frame_ids


def get_directory(dir_name, seq):
    if '{}' in dir_name:
        dir_name = dir_name.format(seq)
    else:
        dir_name = os.path.join(dir_name, seq)
    return dir_name


def propagate_sequence(seq, img_dir, inst_dir, out_dir, cmap, overlap_thrs=0.1, visualize=False):
    inst_dir = get_directory(inst_dir, seq)

    insts = read_detectron2_output(inst_dir)

    cleaned = clean_instances(insts, min_inst_per_frame=2, max_inst_per_frame=2,
                              closing_area=5, overlap_thrs=overlap_thrs, overwrite=not visualize)
    cons, bad_frame_ids = check_consistency(cleaned, overwrite=not visualize)

    if visualize:
        img_dir = get_directory(img_dir, seq)
        imgs = read_detectron2_output(img_dir)
        out = utils.visualization.concat_images([imgs,
                                                 utils.visualization.visualize_superpixels(insts, color_map=cmap),
                                                 utils.visualization.visualize_superpixels(cleaned, color_map=cmap),
                                                 utils.visualization.visualize_superpixels(cons, color_map=cmap)],
                                                n_rows=2)
        utils.io.save_video('{}/{}.avi'.format(out_dir, seq), out)

    utils.io.save_images('{}/{}'.format(out_dir, seq), cons)
    utils.io.save('{}/{}.json'.format(out_dir, seq), bad_frame_ids)

    return seq


def propagate_sequence_wrapper(params):
    return propagate_sequence(*params)


def propagate_sequences(seqs, img_dir, inst_dir, out_dir, overlap_thrs=0.1, visualize=False):
    # Initialization
    cmap = {0: [0, 0, 0],
            1: [0, 255, 0],
            2: [0, 0, 255],
            3: [255, 0, 0],
            4: [127, 127, 0],
            5: [127, 0, 127],
            6: [0, 127, 127],
            7: [127, 127, 127]}

    # Propagate each sequence
    with multiprocessing.Pool() as pool:
        with tqdm(total=len(seqs)) as pbar:
            for seq in pool.imap_unordered(propagate_sequence_wrapper,
                                           zip(seqs,
                                               itertools.repeat(img_dir),
                                               itertools.repeat(inst_dir),
                                               itertools.repeat(out_dir),
                                               itertools.repeat(cmap),
                                               itertools.repeat(overlap_thrs),
                                               itertools.repeat(visualize))):
                print(seq)
                pbar.update()

# TODO __main__
