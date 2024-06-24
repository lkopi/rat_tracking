#!/usr/bin/env python
# coding: utf-8

import copy
from dataclasses import dataclass
from functools import partial
import gc
import itertools
import multiprocessing
import os

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from skimage.measure import label
from skimage.morphology import remove_small_holes
from tqdm import tqdm

import eval_propagation as ep
import kalman_filter
import utils.io
import utils.keypoints
import utils.segmentation
import utils.visualization
import utils.flow
import utils.utils


def read_detectron2_output_parallel(out_dir, read_json=False, segments_from_json=False, multichannel=False,
                                    start_idx=0, n_samples=99999, n_processes=1):
    """Read the results of detectron2"""
    assert not (not read_json and segments_from_json)  # segments_from_json can only be true, if read_json is true

    fns = utils.io.list_directory(out_dir, sort_key=utils.io.filename_to_number,
                                  extension='.png')[start_idx:start_idx + n_samples]

    instances = utils.io.read_arrays_parallel(fns, n_processes=n_processes)
    if multichannel:
        instances[instances == 255] = 0  # TODO: remove (Anna)
        instances = np.round(instances / 127).astype(np.uint8)  # TODO: remove (Anna)
        instances = utils.segmentation.single2multichannel(instances)

    if not read_json:
        return instances
    else:
        jsons = [fn[:-3] + 'json' for fn in fns]
        if not os.path.isfile(jsons[0]):
            print("[WARNING]: json doesn't exists")
            return instances, None

        annots = utils.io.read_files_parallel(jsons, n_processes=n_processes)

        # Load instances from the json file
        if segments_from_json:
            # Initialize instances
            l, h, w = instances.shape[:3]
            if not multichannel:
                n_insts = np.unique(instances)[-1]
                instances = np.zeros((l, h, w, n_insts), dtype=np.uint8)
            else:
                n_insts = instances.shape[-1]
                # instances *= 0  # Shape is already good, set to 0
                instances = np.zeros((*instances.shape[:-1], 12),  # 6),
                                     dtype=np.uint8)  # TODO: make it pretty (count it beforehand)

            mask_shape = (h, w, n_insts)
            with multiprocessing.Pool(processes=n_processes) as pool:
                with tqdm(total=len(fns)) as pbar:
                    for frame_id, instance in enumerate(pool.imap(partial(read_detectron2_json, mask_shape=mask_shape),
                                                                  annots, chunksize=100)):
                        instances[frame_id, ..., :instance.shape[-1]] = instance
                        pbar.update()

            if not multichannel:
                instances = utils.segmentation.multi2singlechannel(instances)
        returns = [instances, annots]
    return tuple(returns)


def read_detectron2_json(annot, mask_shape):
    h, w, n_insts = mask_shape
    instance = np.zeros(mask_shape, dtype=np.uint8)
    for inst_id, anns in annot.items():
        inst_id = int(inst_id)
        for segment in anns['pred_masks']:
            decoded_mask = utils.segmentation.decode_mask(segment, (h, w))
            instance[decoded_mask, inst_id - 1] = inst_id
    return instance


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


def get_instance_ids(frame, return_counts=False):
    inst_ids, counts = np.unique(frame, return_counts=True)
    if inst_ids[0] == 0:
        inst_ids, counts = inst_ids[1:], counts[1:]
    if return_counts:
        return inst_ids, counts
    return inst_ids


@dataclass
class CleanFrameParams:
    remove_tails: bool = True
    ksize: int = 5
    min_area: int = 200

    min_n_inst_per_frame: int = 2  # TODO: eval it with 1 (parameter tuning)
    max_n_inst_per_frame: int = 2

    remove_disproportionate_instances: bool = True
    # max_deviation: float = 0.4
    max_deviation: float = 0.6
    average_area: float = 2300

    # convexity_threshold: float = 0.5  # 0.6
    convexity_threshold: float = 0.4  # 0.6

    keep_only_largest_regions: bool = True
    n_unique_per_class: int = 2

    overlap_threshold: float = 0.1

    return_original: bool = True  # False

    return_scores: bool = True


def find_bbox(np_array, padding=0):
    result = np.where(np_array)
    r1 = max(0, np.min(result[0]) - padding)
    r2 = min(np_array.shape[0], np.max(result[0]) + padding + 1)
    c1 = max(0, np.min(result[1]) - padding)
    c2 = min(np_array.shape[1], np.max(result[1]) + padding + 1)
    return r1, c1, r2, c2


def clean_frame(frame, keypoints=None, params=None, classes=None):
    orig_frame = frame.copy()
    frame = frame.copy()
    scores = []

    if frame.shape[-1] == 0:
        tmp = np.zeros((*frame.shape[:-1], 2), dtype=np.uint8)
        return tmp, tmp, [0] * (params.remove_disproportionate_instances + 0. <
                                params.convexity_threshold + 1 + params.return_original)

    # print('clean_frame', frame.shape, np.unique(frame))
    orig_shape = orig_frame.shape
    multichannel = (frame.ndim == 3)
    tmp = 0 < (orig_frame if not multichannel else orig_frame.sum(axis=-1))
    # (r1, c1), _, _, (r2, c2) = corners(tmp)
    try:
        r1, c1, r2, c2 = find_bbox(tmp, padding=10)
    except:
        print("except", tmp.shape, np.unique(tmp))
        tmp = np.zeros(frame.shape, dtype=np.uint8)
        return tmp, tmp, [0] * (params.remove_disproportionate_instances + 0. <
                                params.convexity_threshold + 1 + params.return_original)
    # print((r1, c1), (r2, c2))
    orig_frame = orig_frame[r1:r2, c1:c2]
    frame = frame[r1:r2, c1:c2]

    if params.remove_tails:
        frame = remove_tails(frame, ksize=params.ksize)
    else:
        frame = smooth_masks(frame, min_area=params.min_area, ksize=params.ksize)
    if params.remove_disproportionate_instances:
        frame, scr = remove_disproportionate_instances(frame, max_deviation=params.max_deviation,
                                                       average_area=params.average_area, return_score=True)
        scores.append(scr)
    if params.convexity_threshold > 0:
        frame, scr = filter_convex_instances(frame, convexity_threshold=params.convexity_threshold,
                                             keep_convex_only=params.remove_tails, return_score=True)
        scores.append(scr)
    if params.keep_only_largest_regions:
        frame = extract_largest_regions(frame)
    if params.overlap_threshold < 1.:
        frame, scr = remove_overlapping_instances(frame, params.overlap_threshold, return_score=True)
        scores.append(scr)
    if params.min_n_inst_per_frame is not None and len(get_instance_ids(frame)) < params.min_n_inst_per_frame:
        frame = np.zeros_like(frame)
    if params.max_n_inst_per_frame is not None:
        frame = keep_n_largest_instances(frame, num_instances=params.max_n_inst_per_frame, classes=classes,
                                         n_unique_per_class=params.n_unique_per_class)
    if params.return_original:
        out_frame = np.zeros_like(orig_frame)
        inst_ids = get_instance_ids(frame)
        for inst_id in inst_ids:
            out_frame[orig_frame == inst_id] = inst_id
        if 0 < len(inst_ids):
            out_frame = smooth_masks(out_frame, min_area=params.min_area, ksize=5)
        # frame = out_frame
        if params.return_scores:
            matched_indices, iou_mat, _, _ = match_proposals(frame, out_frame)

            score = 0
            tmp = iou_mat[matched_indices[0], matched_indices[1]]
            if 0 < len(tmp):
                score = np.average(tmp)
            scores.append(score)
    else:
        out_frame = frame.copy()
        if params.return_scores:
            scores.append(1.)

    tmp = np.zeros(orig_shape, dtype=np.uint8)
    tmp[r1:r2, c1:c2] = out_frame
    out_frame = tmp
    tmp = np.zeros(orig_shape, dtype=np.uint8)
    tmp[r1:r2, c1:c2] = frame
    frame = tmp

    if keypoints is not None:
        out_kps = {}
        inst_ids = get_instance_ids(frame)
        for inst_id in inst_ids:
            out_kps[str(inst_id - 1)] = copy.deepcopy(keypoints[str(inst_id - 1)])
        keypoints = out_kps
        return frame, keypoints

    # if params.return_original:
    if params.return_scores:
        return frame, out_frame, scores
    return frame, out_frame
    # return frame


def keep_n_largest_instances(frame, num_instances=2, classes=None, n_unique_per_class=2):
    frame = frame.copy()

    inst_ids = sort_by_frequency(frame, reverse=True)[1:]  # ignore the background
    if classes is not None:
        n_unique_classes = len(np.unique(classes)[1:])
        n_found_per_class = [0] * n_unique_classes
    if num_instances < len(inst_ids):
        n_selected = 0
        for inst_id in inst_ids:
            if n_selected < num_instances:  # TODO take another look!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if classes is not None:
                    inst_class_id = get_most_common_value(classes[frame == inst_id])
                    if inst_class_id == -1 or n_unique_per_class <= n_found_per_class[inst_class_id - 1]:
                        frame[frame == inst_id] = 0
                    else:
                        n_found_per_class[inst_class_id - 1] += 1
                        n_selected += 1
                else:
                    n_selected += 1
            else:
                frame[frame == inst_id] = 0
    return frame


def remove_disproportionate_instances(frame, max_deviation=0.4, average_area=None, return_score=False):
    frame = frame.copy()
    inst_ids, areas = get_instance_ids(frame, return_counts=True)
    score = 0

    if 1 < len(inst_ids):
        if average_area is not None:
            mean_area = average_area
        else:
            mean_area = np.mean(areas)  # np.median(areas)
        deviation = 1 - np.minimum(areas / mean_area, mean_area / areas)  # TODO: 2021.11.22 switch up
        # deviation = np.abs(areas / mean_area - 1)
        disproportionate_indicies, = np.where(deviation >= max_deviation)

        for idx in disproportionate_indicies:
            frame[frame == inst_ids[idx]] = 0
        tmp = deviation[deviation < max_deviation]
        if 0 < len(tmp):
            score = np.average(tmp) / max_deviation
    else:
        frame = np.zeros_like(frame)

    if return_score:
        return frame, score
    return frame


def remove_overlapping_instances(frame, overlap_threshold=0.5, return_score=False):
    frame = frame.copy()
    inst_ids = get_instance_ids(frame)
    multichannel = (frame.ndim == 3)
    score = 0

    if 1 < len(inst_ids):
        # Calculate IoUs
        all_ious = np.zeros((inst_ids[-1] + 1, inst_ids[-1] + 1))
        for inst_id in inst_ids:
            inst_mask = (frame == inst_id)
            if multichannel:
                inst_mask = inst_mask[..., inst_id - 1]

            ious, label_ids = utils.segmentation.mask_intersection_over_union(frame, inst_mask, use_iou=False)
            all_ious[inst_id, label_ids] = ious
        all_ious = all_ious[1:, 1:]

        np.fill_diagonal(all_ious, 0)
        assert np.all(all_ious[0, :] == all_ious[:, 0]), all_ious
        overlapping_ids, _ = np.where(all_ious >= overlap_threshold)
        overlapping_ids = sort_by_frequency(overlapping_ids, reverse=True, repeat_elements=True)
        for i in overlapping_ids[:len(overlapping_ids) // 2]:
            frame[frame == i + 1] = 0

        tmp = all_ious[all_ious < overlap_threshold]
        if 0 < len(tmp):
            score = 1 - np.average(tmp) / overlap_threshold

    if return_score:
        return frame, score
    return frame


def sort_by_frequency(array, reverse=False, repeat_elements=False, return_frequency=False):
    unique_elements, frequency = np.unique(array, return_counts=True)
    sorted_indexes = np.argsort(frequency)
    if reverse:
        sorted_indexes = sorted_indexes[::-1]
    sorted_by_freq = unique_elements[sorted_indexes]

    if repeat_elements:
        sorted_by_freq = np.repeat(sorted_by_freq, frequency[sorted_indexes])

    if not return_frequency:
        return sorted_by_freq
    return sorted_by_freq, frequency[sorted_indexes]


def extract_largest_regions(frame):
    frame = frame.copy()
    inst_ids = get_instance_ids(frame)
    multichannel = (frame.ndim == 3)

    for inst_id in inst_ids:
        inst_mask = (frame == inst_id)
        if multichannel:
            inst_mask = inst_mask[..., inst_id - 1]

        largest_segment = utils.segmentation.extract_largest_object(inst_mask)

        if not multichannel:
            frame[inst_mask] = 0
            frame[largest_segment] = inst_id
        else:
            frame[inst_mask, inst_id - 1] = 0
            frame[largest_segment, inst_id - 1] = inst_id

    return frame


def filter_convex_instances(frame, convexity_threshold=0.6, keep_convex_only=False, return_score=False):
    frame = frame.copy()
    inst_ids = get_instance_ids(frame)
    multichannel = (frame.ndim == 3)
    score, n_scores = 0, 0

    for inst_id in inst_ids:
        inst_mask = (frame == inst_id)
        if multichannel:
            inst_mask = inst_mask[..., inst_id - 1]

        is_convex, overlap = utils.segmentation.is_convex(inst_mask, threshold=convexity_threshold, return_overlap=True)

        if keep_convex_only:
            is_convex = not is_convex
        if is_convex:
            if not multichannel:
                frame[inst_mask] = 0
            else:
                frame[inst_mask, inst_id - 1] = 0
        else:
            score += overlap
            n_scores += 1

    if 0 < n_scores:
        score /= n_scores
    if return_score:
        return frame, score
    return frame


def smooth_masks(frame, min_area=200, ksize=10):
    frame = frame.copy()

    multichannel = (frame.ndim == 3)
    if not multichannel:
        frame = frame[..., None]

    for inst_id in range(frame.shape[-1]):
        frame[..., inst_id] = remove_small_regions(frame[..., inst_id], bg_label=0, min_area=min_area)
        if 0 < ksize and 0 < np.count_nonzero(frame[..., inst_id]):
            frame[..., inst_id] = utils.segmentation.apply_closing(frame[..., inst_id], ksize=ksize)
            frame[..., inst_id] = fill_small_holes(frame[..., inst_id], area_threshold=min_area)

    if not multichannel:
        frame = frame[..., 0]

    return frame


def remove_tails(frame, ksize=5):
    frame = frame.copy()
    if ksize == 0:
        return frame

    multichannel = (frame.ndim == 3)
    if not multichannel:
        frame = frame[..., None]

    for inst_id in range(frame.shape[-1]):
        frame[..., inst_id] = utils.segmentation.apply_opening(frame[..., inst_id], ksize=ksize)

    if not multichannel:
        frame = frame[..., 0]

    return frame


def preprocess_instances_parallel(instances, params, cons_dir, orig_dir, start_idx=0,
                                  n_processes=-1, save_only_scores=False):
    # assert params.return_original and params.return_scores
    if n_processes == -1:
        n_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=min(n_processes, len(instances))) as pool:
        with tqdm(total=len(instances)) as pbar:
            for idx, frame in enumerate(pool.imap(partial(clean_frame, keypoints=None, params=params, classes=None),
                                                  instances)):
                if not save_only_scores:
                    utils.io.save_multichannel_images(cons_dir, [frame[0]], extension='.png', start_idx=start_idx + idx)
                    utils.io.save_multichannel_images(orig_dir, [frame[1]], extension='.png', start_idx=start_idx + idx)
                utils.io.save_files(cons_dir, [frame[2]], extension='.pkl', start_idx=start_idx + idx)
                pbar.update()


def get_most_common_value(mask, bg_value=0, return_area=False, return_coverage=False):
    ids, areas = sort_by_frequency(mask, reverse=True, return_frequency=True)
    if ids[0] == bg_value:
        ids = ids[1:]
        areas = areas[1:]

    output = [-1] * (1 + return_area + return_coverage)
    if 0 < len(ids):
        output[0] = ids[0]
        next_idx = 1
        if return_area:
            output[next_idx] = areas[0]
            next_idx += 1
        if return_coverage:
            output[next_idx] = areas[0] / np.count_nonzero(mask)

    return tuple(output) if 1 < len(output) else output[0]


@dataclass
class ScoreMatrixParams:
    use_distance: bool = True
    predicted_positions: list = None
    guides: (np.ndarray, np.ndarray) = None
    classes: (np.ndarray, np.ndarray) = None
    instance_ids_and_scores: np.ndarray = None
    class_ids_and_scores: np.ndarray = None
    use_iou: bool = True
    take_smaller_iou_value: bool = False


def match_proposals(curr_masks, next_masks, iou_thrs=0., params=None):
    iou_mat, curr_ids, next_ids = calculate_score_matrix(curr_masks, next_masks,
                                                         params=params)
    matched_indices = linear_assignment(-iou_mat)
    if 0. < iou_thrs:
        index_mask = (iou_mat[matched_indices[0], matched_indices[1]] >= iou_thrs)
        matched_indices = (matched_indices[0][index_mask], matched_indices[1][index_mask])
    return matched_indices, iou_mat, curr_ids, next_ids


def calculate_score_matrix(curr_masks, next_masks, params=None):
    if params is None:
        params = ScoreMatrixParams()
    norm_value = 1 + params.use_distance + (params.guides is not None) + (params.classes is not None)
    multichannel = (curr_masks.ndim == 3)
    confidence_weight = 500

    curr_ids, curr_areas = np.unique(curr_masks, return_counts=True)
    curr_ids, curr_areas = curr_ids[1:], curr_areas[1:]
    next_ids, next_areas = np.unique(next_masks, return_counts=True)
    next_ids, next_areas = next_ids[1:], next_areas[1:]
    score_mat = np.zeros((len(curr_ids), len(next_ids)), dtype=np.float32)
    for t, tid in enumerate(curr_ids):
        for d, did in enumerate(next_ids):
            if multichannel:
                cmask, nmask = curr_masks[..., tid - 1] == tid, next_masks[..., did - 1] == did
            else:
                cmask, nmask = curr_masks == tid, next_masks == did
            score_mat[t, d] += \
                utils.segmentation.mask_intersection_over_union(cmask, nmask, use_iou=params.use_iou,
                                                                take_smaller=params.take_smaller_iou_value)[0]
                # 2 * ...
            if params.use_distance:
                if params.predicted_positions is None:
                    cx, cy = utils.keypoints.calculate_weighted_center(cmask)
                else:
                    cx, cy = params.predicted_positions[t]
                nx, ny = utils.keypoints.calculate_weighted_center(nmask)
                score_mat[t, d] += 1 / max(1, np.sqrt((cx - nx) ** 2 + (cy - ny) ** 2))
            if params.guides is not None:
                if params.instance_ids_and_scores is not None:
                    curr_id, curr_coverage = params.instance_ids_and_scores[tid - 1, :]
                else:
                    curr_id, curr_coverage = get_most_common_value(params.guides[0][cmask], return_area=True)
                curr_coverage /= curr_areas[t]

                next_id, next_coverage = get_most_common_value(params.guides[1][nmask], return_area=True)
                next_coverage /= next_areas[d]

                if curr_coverage * next_coverage > 0:
                    score_mat[t, d] += confidence_weight * (curr_coverage * next_coverage) * (curr_id == next_id)
            if params.classes is not None:
                if params.class_ids_and_scores is not None:
                    curr_id, curr_coverage = params.class_ids_and_scores[tid - 1, :]
                else:
                    curr_id, curr_coverage = get_most_common_value(params.classes[0][cmask], return_area=True)
                curr_coverage /= curr_areas[t]

                next_id, next_coverage = get_most_common_value(params.classes[1][nmask], return_area=True)
                next_coverage /= next_areas[d]

                if curr_coverage * next_coverage > 0:
                    score_mat[t, d] += confidence_weight * (curr_coverage * next_coverage) * (curr_id == next_id)

    score_mat /= norm_value
    return score_mat, curr_ids, next_ids


def propagate_with_optical_flow(flow, masks, classes=None, use_avg=False, swap_uv_channels=True):
    is_multichannel = utils.segmentation.is_multichannel(masks[None])

    if swap_uv_channels:
        flow = flow[..., ::-1]
    if use_avg:
        segments = masks
        if not is_multichannel:
            segments = utils.segmentation.single2multichannel(masks[None])[0]
        avg_flow = np.zeros_like(flow)
        avg_flow[..., 0] = np.around(utils.flow.get_average_flow(flow[..., 0], segments))
        avg_flow[..., 1] = np.around(utils.flow.get_average_flow(flow[..., 1], segments))
        flow = avg_flow
    else:
        flow = flow.astype(np.int32)

    propagated_masks = utils.flow.warp_image(masks, flow, is_relative=True)

    if classes is not None:
        propagated_classes = utils.flow.warp_image(masks, flow, is_relative=True)
        return propagated_masks, propagated_classes
    return propagated_masks


def _read_pred(fn):
    return utils.io.read_multichannel_images([fn])[0]


def _write_pred(out_dir, idx, pred):
    return utils.io.save_multichannel_images(out_dir, [pred], start_idx=idx)


def _read_of(fn):
    return utils.io.read_arrays([fn], shape=(1, 420, 640, 2), use_cropping=True, use_mmcv=True)[0]


class Trajectory2D:
    def __init__(self, n_instances):
        dt = 0.05
        u = 0
        std_acc = 10   # we assume that the standard deviation of the acceleration is 0.25 (m/s^2)
        std_meas = 10  # and standard deviation of the measurement is 1.2 (m)

        self.n_instances = n_instances
        self.measurements, self.predictions = [], []
        self.kfs = []
        for _ in range(self.n_instances):
            self.measurements.append([])
            self.predictions.append([])
            self.kfs.append(kalman_filter.KalmanFilter(dt, u, std_acc, std_meas))
            self.kfs[-1].predict()

    def update(self, coords):
        assert len(coords) == self.n_instances, coords

        for inst_id, coord in enumerate(coords):
            invalid = coord is None or len(coord) == 0
            if not invalid:
                self.measurements[inst_id].append(coord)
                self.kfs[inst_id].update(coord)
            prediction = self.kfs[inst_id].predict()[0].tolist()[0]
            self.predictions[inst_id].append(prediction)
            if invalid:
                self.measurements[inst_id].append(prediction)

    @property
    def get_latest_prediction(self):
        out = [None]*self.n_instances
        for inst_id in range(self.n_instances):
            out[inst_id] = self.predictions[inst_id][-1]
        return out

    @property
    def get_predictions(self):
        return np.asarray(self.predictions)

    @property
    def get_measurements(self):
        return np.asarray(self.measurements)

    def __len__(self):
        return len(self.measurements[0])


def propagate_predictions_memory_conservative(pred_fns, n_instances=2, iou_thrs=0.,
                                              of_fns=None, reverse=False, initial_frame_idx=None, orig_fns=None,
                                              read_frame=_read_pred, write_frame=_write_pred, read_of=_read_of,
                                              out_dir=None, use_trajectories=True):
    """
    Propagate (and relabel) predictions based on the consistency with the previous frame's prediction.
    Use the previous prediction if it cannot find a proper proposal.
    """
    if out_dir is None:
        out_dir = os.path.dirname(pred_fns[0])

    # Find the first frame where all instances are present
    if initial_frame_idx is None:
        initial_frame_idx = 0
        while initial_frame_idx < len(pred_fns) \
                and len(np.unique(read_frame(pred_fns[initial_frame_idx]))) != n_instances + 1:
            initial_frame_idx += 1
        if initial_frame_idx == len(pred_fns):
            return list(range(len(pred_fns))), np.zeros((n_instances, len(pred_fns), 2))

    # If the initial frame is not the first frame, then propagate backwards
    bad_frame_ids = []
    trajectories = Trajectory2D(n_instances)
    if initial_frame_idx != 0 and not reverse:
        bad_frame_ids, positions = propagate_predictions_memory_conservative(pred_fns,
                                                                             n_instances=n_instances,
                                                                             iou_thrs=iou_thrs,
                                                                             of_fns=of_fns, reverse=True,
                                                                             initial_frame_idx=initial_frame_idx,
                                                                             orig_fns=orig_fns,
                                                                             read_frame=read_frame,
                                                                             write_frame=write_frame,
                                                                             read_of=read_of,
                                                                             out_dir=out_dir,
                                                                             use_trajectories=use_trajectories)
        # Update the trajectories
        for i in range(len(positions[0]) - 1, -1, -1):
            trajectories.update(positions[:, i])
        bad_frame_ids = bad_frame_ids[::-1]

    # Initialize the propagation
    prev_pred = read_frame(pred_fns[initial_frame_idx])
    channel_ids = np.unique(prev_pred)[1:] - 1
    coords = utils.keypoints.heatmap2keypoints(prev_pred[..., channel_ids], round_coords=False)
    trajectories.update(coords)
    if orig_fns is not None:
        prev_orig = read_frame(orig_fns[initial_frame_idx])
        if not reverse:
            write_frame(out_dir, initial_frame_idx, prev_orig)
    if of_fns is not None and initial_frame_idx < len(of_fns):
        of = read_of(of_fns[initial_frame_idx if not reverse else max(0, initial_frame_idx - 1)])
        if reverse:     # Rough estimate of the backward optical flow
            of *= -1
        prev_pred = propagate_with_optical_flow(of, prev_pred)

    # Propagate forward
    idx_range = range(initial_frame_idx + 1, len(pred_fns)) if not reverse else range(initial_frame_idx - 1, -1, -1)
    for idx in idx_range:
        # Read the current prediction
        curr_pred = read_frame(pred_fns[idx])
        out_pred = np.zeros_like(curr_pred)
        if orig_fns is not None:
            curr_orig = read_frame(orig_fns[idx])
            out_orig = np.zeros_like(curr_orig)

        # Match the previous and the current predictions
        params = ScoreMatrixParams()
        if use_trajectories:
            params.predicted_positions = trajectories.get_latest_prediction
        matched_indices, iou_mat, curr_ids, next_ids = match_proposals(prev_pred, curr_pred,
                                                                       iou_thrs=iou_thrs, params=params)

        # Relabel the current instances based on the matches
        track_map = dict(zip(next_ids[matched_indices[1]].tolist(), curr_ids[matched_indices[0]].tolist()))
        for k in track_map.keys():
            out_pred[curr_pred[..., k - 1] == k, track_map[k] - 1] = track_map[k]
            if orig_fns is not None:
                out_orig[curr_orig[..., k - 1] == k, track_map[k] - 1] = track_map[k]
        # If no match found for some instances, preserve their previous labels, and mark them as bad
        # TODO: alternative solution: interpolate with pixel-wise OF propagation
        kept_instances = []
        if len(track_map) < n_instances:
            keep_map = dict(zip(list(set(curr_ids) - set(track_map.keys())),
                                list(set(curr_ids) - set(track_map.values()))))
            for k in keep_map.keys():
                out_pred[prev_pred == keep_map[k]] = keep_map[k]
                if orig_fns is not None:
                    out_orig[prev_orig == keep_map[k]] = keep_map[k]
                kept_instances.append(keep_map[k])
            bad_frame_ids.append(idx)

        # Prepare the next iteration
        prev_pred = out_pred

        # Update trajectories
        coords = utils.keypoints.heatmap2keypoints(prev_pred[..., channel_ids], round_coords=False)
        if 0 < len(kept_instances):
            kept_instances = set(kept_instances)
            for inst_id, channel_id in enumerate(channel_ids):
                if channel_id in kept_instances:
                    coords[inst_id] = None
        trajectories.update(coords)

        if orig_fns is not None:
            prev_orig = out_orig
            write_frame(out_dir, idx, out_orig)
        else:
            write_frame(out_dir, idx, out_pred)

        if of_fns is not None and idx < len(of_fns):
            of = read_of(of_fns[idx if not reverse else max(0, idx - 1)])
            if reverse:     # Rough estimate of the backward optical flow
                of *= -1
            prev_pred = propagate_with_optical_flow(of, prev_pred)

    return bad_frame_ids, trajectories.get_predictions


def get_directory(dir_name, seq):
    if '{}' in dir_name:
        dir_name = dir_name.format(seq)
    else:
        dir_name = os.path.join(dir_name, seq)
    return dir_name


def select_subsequent_frame_ids(frame_ids):
    frame_ids = np.asarray(frame_ids)
    diffs = frame_ids[1:] - frame_ids[:-1]
    start_indices, = np.where(diffs > 1)
    start_indices = np.concatenate(([0], start_indices + 1), axis=0)

    start_frames = frame_ids[start_indices]
    end_frames = np.concatenate((frame_ids[start_indices[1:] - 1], frame_ids[-1:]), axis=0)
    lengths = end_frames - start_frames + 1

    return start_frames, end_frames, lengths


def determine_upper_rat(predictions):
    # ?idea: always take the latest/newest instance at each position as upper. => then take the average through time

    # Determine instance (channel) ids
    inst_ids = []
    for ch in range(predictions.shape[-1]):
        if np.any(predictions[..., ch] > 0):
            inst_ids.append(ch)
    if len(inst_ids) < 2:
        return np.zeros((predictions.shape[:3]), dtype=np.uint8)
    assert len(inst_ids) == 2, inst_ids

    updates = np.concatenate((predictions[:1],
                              np.abs(predictions[1:].astype(np.float16) - predictions[:-1]).astype(np.uint8)),
                             axis=0).astype(np.uint8)

    intersections = np.zeros((predictions.shape[:3]), dtype=np.uint8)
    intersections[predictions[..., inst_ids[0]] > 0] += 1
    intersections[predictions[..., inst_ids[1]] > 0] += 1

    updates[intersections < 2, :] = 0
    del intersections

    counts = np.zeros((len(predictions), predictions.shape[-1]), dtype=np.uint8)
    for i in range(len(predictions)):
        for inst_id in inst_ids:
            counts[i, ..., inst_id] = np.count_nonzero(updates[i, ..., inst_id] > 0)
    del updates
    gc.collect()

    # TODO temporal smoothing
    cmax = np.max(counts)
    if cmax != 0:
        tmp = np.sum(counts, axis=1)
        updated_frame_ids, = np.where(tmp == 0)
        if 0 < len(updated_frame_ids):
            start_frame_ids, end_frame_ids, _ = select_subsequent_frame_ids(updated_frame_ids)
            for start, end in zip(list(start_frame_ids), list(end_frame_ids)):
                nearest = start - 1
                if nearest < 0:
                    nearest = end + 1
                    if len(predictions) <= nearest:
                        continue
                curr_count = counts[nearest]
                for idx in range(start, end + 1):
                    counts[idx] = curr_count

    updated_instances = np.argmax(counts, axis=1)
    out_predictions = np.zeros(predictions.shape[:3], dtype=np.uint8)
    for i in range(len(predictions)):
        tmp = set(inst_ids)
        upper_inst_id = updated_instances[i]
        if upper_inst_id not in tmp:
            upper_inst_id = tmp.pop()
        else:
            tmp.remove(upper_inst_id)
        lower_inst_id = tmp.pop()
        for inst_id in (lower_inst_id, upper_inst_id):
            mask = predictions[i, ..., inst_id] > 0
            out_predictions[i, mask] = inst_id + 1

    return out_predictions


def match_label_ids(preds, gts, keypoints=None, return_bool=False):
    pred_ids = np.unique(preds)[1:]
    gt_ids, gt_areas = np.unique(gts, return_counts=True)
    gt_ids = gt_ids[1:]
    # print(preds.shape, gts.shape, pred_ids, gt_ids)

    if len(gt_ids) == 0:
        print("[WARNING] gts is empty")
        if return_bool:
            return preds, False
        return preds
    if len(pred_ids) == 1:
        pid = pred_ids[0]
        pred_ids = pred_ids.tolist()
        pred_ids.append(gt_ids[0] if pid != gt_ids[0] else gt_ids[1])
        pred_ids = np.asarray(pred_ids)
    if len(gt_ids) == 1:
        pid = gt_ids[0]
        gt_ids = gt_ids.tolist()
        gt_ids.append(pred_ids[0] if pid != pred_ids[0] else pred_ids[1])
        gt_ids = np.asarray(gt_ids)
    assert len(pred_ids) == 2, pred_ids
    assert len(gt_ids) == 2, gt_ids

    iou_orig = ep.calculate_iou(gts == gt_ids[0],
                                preds == pred_ids[0]) + ep.calculate_iou(gts == gt_ids[1], preds == pred_ids[1])
    iou_switched = ep.calculate_iou(gts == gt_ids[0],
                                    preds == pred_ids[1]) + ep.calculate_iou(gts == gt_ids[1], preds == pred_ids[0])

    switched = iou_switched > iou_orig
    if switched:
        tmp = np.zeros_like(preds)
        tmp[preds == pred_ids[0]] = pred_ids[1]
        tmp[preds == pred_ids[1]] = pred_ids[0]
        preds = tmp

        if keypoints is not None:
            for i in range(len(keypoints)):
                try:
                    keypoints[i]['0'], keypoints[i]['1'] = keypoints[i]['1'], keypoints[i]['0']
                except:
                    if len(keypoints[i]) > 0:
                        orig_key = int(list(keypoints[i].keys())[0])
                        new_key = 1 - orig_key
                    print('ERR', i, keypoints[i].keys())
    outs = [preds]
    if keypoints is not None:
        outs.append(keypoints)
    if return_bool:
        outs.append(switched)
    return outs[0] if len(outs) == 1 else tuple(outs)


def calculate_average_size(instances):
    avg_size = 0
    for c in instances:
        _, areas = get_instance_ids(c, return_counts=True)
        mean = np.mean(areas)
        if 0 < mean:
            avg_size += mean
    avg_size /= len(instances)
    return avg_size


def propagate_predictions_memory_conservative_wrapper(params):
    out_dir, orig_dir, of_dir, chunksize = params

    cons_fns = utils.io.list_directory(out_dir, sort_key=utils.io.filename_to_number, extension='.json')
    temp_fns = utils.io.list_directory(out_dir, extension='.png')
    temp_fns += utils.io.list_directory(out_dir, extension='.json')
    orig_fns = utils.io.list_directory(orig_dir, sort_key=utils.io.filename_to_number, extension='.json')
    of_fns = utils.io.list_directory(of_dir, sort_key=utils.io.filename_to_number)

    bad_frame_ids, positions = propagate_predictions_memory_conservative(cons_fns,
                                                                         of_fns=of_fns, orig_fns=orig_fns,
                                                                         use_trajectories=True)
    utils.io.remove(orig_dir)
    print('[FINISHED]: propagated predictions', out_dir)

    # chunksize = 300
    for i in tqdm(range(0, len(cons_fns), chunksize), total=(len(cons_fns) + chunksize - 1) // chunksize):
        outs = utils.visualization.visualize_predictions(
            determine_upper_rat(utils.io.read_multichannel_images(cons_fns[i:i + chunksize])))
        utils.io.save_images(out_dir, outs, start_idx=i)
    print('[FINISHED]: influence', out_dir)
    utils.io.remove_files(temp_fns)
    utils.io.save(out_dir + '.json', bad_frame_ids)
    utils.io.save(out_dir + '_positions.txt', positions.astype(float).tolist())
    return 'DONE'


def propagate_sequences_memory_conservative(seqs, inst_dir_base, out_dir, overlap_thrs=0.1,
                                            params=None, of_dir_base=None, chunksize=9999,
                                            save_only_scores=False):
    if params is None:
        params = CleanFrameParams(overlap_threshold=overlap_thrs)  # , return_original=True)
    else:
        params = CleanFrameParams(**params)
    print(params)
    n_processes = len(seqs)

    out_dirs = []
    orig_dirs = []
    of_dirs = []
    print('[START]: pre-processing')
    skipped = False
    for seq in seqs:
        print(seq)

        if of_dir_base is not None:
            of_dirs.append(get_directory(of_dir_base, seq))
        out_dirs.append('{}/{}'.format(out_dir, seq))
        orig_dirs.append('{}/{}_orig'.format(out_dir, seq))

        if os.path.exists(out_dirs[-1]) and os.path.exists(orig_dirs[-1]):
            skipped = True
            continue
        inst_dir = get_directory(inst_dir_base, seq)
        curr_idx, finished = 0, False
        while not finished:
            # insts, _, = read_detectron2_output_parallel(inst_dir, multichannel=True,
            insts = read_detectron2_output_parallel(inst_dir, multichannel=True,  # TODO: remove (Anna)
                                                    read_json=False, segments_from_json=False,  # TODO: remove (Anna)
                                                    # read_json=True, segments_from_json=True,
                                                    start_idx=curr_idx, n_samples=chunksize,
                                                    n_processes=n_processes)
            preprocess_instances_parallel(insts, params, out_dirs[-1], orig_dirs[-1],
                                          start_idx=curr_idx, n_processes=n_processes,
                                          save_only_scores=save_only_scores)
            finished = len(insts) < chunksize
            curr_idx += chunksize
    print('[FINISHED]: pre-processing')
    if not skipped:
        del insts
        gc.collect()
    if save_only_scores:
        return

    print('[START]: propagation')
    pool = multiprocessing.Pool(processes=n_processes)
    with tqdm(total=len(seqs)) as pbar:
        for _ in pool.imap_unordered(propagate_predictions_memory_conservative_wrapper,
                                     zip(out_dirs, orig_dirs, of_dirs, itertools.repeat(chunksize))):
            pbar.update()
    pool.close()
    pool.join()


def match_label_ids_wrapper(params):
    seq, mask_dir, guide_dir = params

    masks = utils.io.read_arrays(
        utils.io.list_directory(get_directory(mask_dir, seq), sort_key=utils.io.filename_to_number, extension='.png'))
    # masks[masks == 255] = 0  # TODO: remove (Anna)
    # masks = np.round(masks / 127).astype(np.uint8)  # TODO: remove (Anna)
    if utils.segmentation.is_multichannel(masks):
        masks = utils.segmentation.multi2singlechannel(masks)

    guides = utils.io.read_arrays(
        utils.io.list_directory(get_directory(guide_dir, seq), sort_key=utils.io.filename_to_number, extension='.png'))
    if utils.segmentation.is_multichannel(guides):
        guides = utils.segmentation.multi2singlechannel(guides)
    if np.any(guides >= 127):
        guides //= 127

    return seq, match_label_ids(masks, guides)


def match_labels(mask_dir, guide_dir, out_dir, n_processes=-1):
    seqs = utils.io.list_directory(guide_dir, only_dirs=True, full_path=False)

    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, len(seqs))

    # Propagate each sequence
    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm(total=len(seqs)) as pbar:
            for seq, matched_masks in pool.imap_unordered(match_label_ids_wrapper,
                                                          zip(seqs,
                                                              itertools.repeat(mask_dir),
                                                              itertools.repeat(guide_dir))):
                utils.io.save_images(get_directory(out_dir, seq), matched_masks)
                fns = utils.io.list_directory(get_directory(mask_dir, seq), extension='.pkl')
                for fn in fns:
                    utils.io.copy_file(fn, fn.replace(mask_dir, out_dir))
                pbar.update()


def determine_sequences_to_switch_up(switched_sequences, all_sequences):
    is_switched = np.zeros(len(all_sequences), dtype=np.bool)
    if switched_sequences is not None:
        switched_sequences = set(utils.io.read(switched_sequences)
                                 if isinstance(switched_sequences, str) else switched_sequences)
        for idx, seq in enumerate(all_sequences):
            if seq in switched_sequences:
                is_switched[idx:] = ~is_switched[idx:]
    seqs_to_switch_up = [all_sequences[i] for i in range(len(all_sequences)) if is_switched[i]]
    return seqs_to_switch_up


def match_label_ids_between_sequences(pred_dir):
    seqs = utils.io.list_directory(pred_dir, only_dirs=True, full_path=False)
    switched_sequences = set()

    for s1, s2 in zip(seqs, seqs[1:]):
        s1_fn = utils.io.list_directory(get_directory(pred_dir, s1), sort_key=utils.io.filename_to_number,
                                        extension='.png')[-1]
        s2_dir = get_directory(pred_dir, s2)
        s2_fns = utils.io.list_directory(s2_dir, sort_key=utils.io.filename_to_number, extension='.png')
        masks = utils.io.read_arrays([s1_fn, s2_fns[0]])
        multichannel = utils.segmentation.is_multichannel(masks)
        if multichannel:
            masks = utils.segmentation.multi2singlechannel(masks)
        mask1, mask2 = masks[0], masks[1]
        print('Matching {} and {}'.format(s1, s2))
        _, switched = match_label_ids(mask1, mask2, return_bool=True)
        if switched:
            # if s1 not in switched_sequences:
            switched_sequences.add(s2)

    seqs_to_switch_up = determine_sequences_to_switch_up(switched_sequences, seqs)
    return seqs_to_switch_up
