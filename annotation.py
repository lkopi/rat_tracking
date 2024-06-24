#!/usr/bin/env python
# coding: utf-8

from dataclasses import dataclass
import itertools
import math
import multiprocessing

import numpy as np

import utils
import propagate_instances as pi
import eval_propagation as eval
import utils.io
import utils.segmentation
import utils.visualization

cmap = {0: [0, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 0, 0],
        4: [127, 127, 0],
        5: [127, 0, 127],
        6: [0, 127, 127],
        7: [127, 127, 127]}

# python annotation.py

# exp_name = "2021122_07_temp_iou"
exp_name = "2021122_06_temp_ACD"
# exp_name = "2021122_08_temp"
def run():
    # exp_name = "2021122_04_temp_iou_scores"
    # exp_name = "2021122_05_temp_iou_AD"
    # exp_name = "2021122_06_temp_D"
    gt_base_dir = "/media/hdd2/lkopi/datasets/rats_clean/test_videos/masks"
    of_base_dir = "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images/{}/GMA"

    mask_base_dir_list = [
        "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched"
    ]

    sequences = utils.io.list_directory(gt_base_dir, only_dirs=True, full_path=False)  #[1:]

    for mask_base_dir in mask_base_dir_list:
        confidence_threshold = 0.4
        for confidence_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            out_dir = '{}_conf_thres{}_{}'.format(mask_base_dir, confidence_threshold, exp_name)
            # sequences = ['hard_2_fin']
            print(sequences)
            n_annotated_frames = 0
            orig_track_switches, fixed_track_switches = 0, 0
            orig_avg_iou, fixed_avg_iou = 0., 0.

            # for seq in sequences:
            #     print(seq)
            #     _orig_track_switches, _orig_avg_iou, _fixed_track_switches, _fixed_avg_iou, _n_annotated_frames = \
            #         process_sequence(mask_base_dir, gt_base_dir, of_base_dir, seq, out_dir, confidence_threshold)

            with multiprocessing.Pool() as pool:
                for _orig_track_switches, _orig_avg_iou, _fixed_track_switches, _fixed_avg_iou, _n_annotated_frames\
                        in pool.imap_unordered(process_sequence_wrapper,
                                               zip(itertools.repeat(mask_base_dir),
                                                   itertools.repeat(gt_base_dir),
                                                   itertools.repeat(of_base_dir),
                                                   sequences,
                                                   itertools.repeat(out_dir),
                                                   itertools.repeat(confidence_threshold))):

                    orig_track_switches += _orig_track_switches
                    orig_avg_iou += _orig_avg_iou
                    fixed_track_switches += _fixed_track_switches
                    fixed_avg_iou += _fixed_avg_iou
                    n_annotated_frames += _n_annotated_frames

            results = {
                'orig': {
                    'ts': orig_track_switches,
                    'iou': orig_avg_iou / len(sequences)
                },
                'fixed': {
                    'ts': fixed_track_switches,
                    'iou': fixed_avg_iou / len(sequences)
                },
                'confidence_threshold': confidence_threshold,
                'n_annotated_frames': n_annotated_frames,
                'input_dir': mask_base_dir
            }
            utils.io.save('{}/summary.json'.format(out_dir), results)


def process_sequence_wrapper(params):
    return process_sequence(*params)


def process_sequence(mask_base_dir, gt_base_dir, of_base_dir, seq, out_dir, confidence_threshold,
                     frame_idx=None, params=None):
    marked_frames, masks, gts, ofs, scores = load_annotations(mask_base_dir, gt_base_dir, of_base_dir, seq)

    n_track_switches, avg_iou = eval.evaluate_prediction(gts, masks)
    orig_track_switches = n_track_switches
    orig_avg_iou = avg_iou
    vid_fn = '{}/{}_ts{}_iou{}.avi'.format(out_dir, seq, n_track_switches, avg_iou)
    utils.io.save_video(vid_fn, utils.visualization.visualize_superpixels(masks, color_map=cmap))
    print(vid_fn)

    n_annotated_frames = 0
    while 0 < len(marked_frames):
        confidence_scores, nearest_frames = calculate_confidence_scores(marked_frames, masks, scores, params=params)
        if np.min(confidence_scores) > confidence_threshold:
            print(f"Finished {seq} with confidence of {np.min(confidence_scores)}")
            break

        masks, marked_frames, neig_start_idx, worst_frame_idx, neig_end_idx = annotate_worst_prediction(
            confidence_scores, marked_frames, nearest_frames, masks, gts, ofs, frame_idx=frame_idx)
        n_annotated_frames += 1

        n_track_switches, avg_iou = eval.evaluate_prediction(gts, masks)
        vid_fn = '{}/{}_iter{}_{}-{}-{}_ts{}_iou{}.avi'.format(out_dir, seq, n_annotated_frames,
                                                               neig_start_idx, worst_frame_idx,
                                                               neig_end_idx, n_track_switches, avg_iou)
        utils.io.save_video(vid_fn, utils.visualization.visualize_superpixels(masks, color_map=cmap))
        print(vid_fn)
        frame_idx = None
        #if frame_idx is not None:
        #    break

    fixed_track_switches = n_track_switches
    fixed_avg_iou = avg_iou

    return orig_track_switches, orig_avg_iou, fixed_track_switches, fixed_avg_iou, n_annotated_frames


def annotate_worst_prediction(confidence_scores, marked_frames, nearest_frames, masks, gts, ofs, frame_idx=None):
    neighborhood, worst_frame_idx, neig_start_idx, neig_end_idx, worst_marked_idx, neig_ofs = select_neighboring_predictions(confidence_scores, marked_frames, nearest_frames, masks, ofs, frame_idx=frame_idx)
    neighborhood[worst_frame_idx - neig_start_idx] = gts[worst_frame_idx]

    backward_prop, forward_prop, marked_frame_idx = propagate_masks(neighborhood, worst_frame_idx, neig_start_idx, neig_ofs)

    fixed_masks, remaining_marked_frames = update_predictions(masks, forward_prop, backward_prop, worst_frame_idx, neig_start_idx, neig_end_idx, marked_frames, worst_marked_idx, ofs)
    if marked_frame_idx != -1:
        remaining_marked_frames.append(marked_frame_idx)
        remaining_marked_frames = list(set(remaining_marked_frames))

    return fixed_masks, remaining_marked_frames, neig_start_idx, worst_frame_idx, neig_end_idx


def update_predictions(masks, forward_prop, backward_prop, worst_frame_idx, neig_start_idx, neig_end_idx, marked_frames, worst_marked_idx, ofs):
    fixed_masks = masks.copy()
    print(neig_start_idx,worst_frame_idx,neig_end_idx, len(backward_prop), len(forward_prop))
    fixed_masks[neig_start_idx:worst_frame_idx] = backward_prop[:-1]  # [:worst_frame_idx - neig_start_idx]
    fixed_masks[worst_frame_idx:neig_end_idx] = forward_prop  # [worst_frame_idx - neig_start_idx:]

    remaining_marked_frames = marked_frames[:worst_marked_idx] + marked_frames[worst_marked_idx + 1:]

    # Check overall consistency
    read_pred = lambda x: x
    write_pred = lambda x,y,z: None
    fixed_masks = utils.segmentation.single2multichannel(fixed_masks)
    fixed_masks = pi.propagate_predictions_memory_conservative(fixed_masks, multichannel=True, of_fns=ofs,
                                                               read_frame=read_pred, read_of=read_pred,
                                                               write_frame=write_pred,
                                                               out_dir='./nothing_should_be_here',
                                                               return_preds=True)[1]
    fixed_masks = utils.segmentation.multi2singlechannel(fixed_masks)
    '''
    if 0 < neig_start_idx:
        _, bad_frames = pi.check_consistency(fixed_masks[neig_start_idx-1:neig_start_idx+1].copy())
        if 0 < len(bad_frames):
            remaining_marked_frames.append(neig_start_idx)
            print('fixed prev', worst_frame_idx)

    if neig_end_idx < len(masks):
        _, bad_frames = pi.check_consistency(fixed_masks[neig_end_idx-1:neig_end_idx+1].copy())
        if 0 < len(bad_frames):
            remaining_marked_frames.append(neig_end_idx)
            print('fixed next', worst_frame_idx)
    '''

    return fixed_masks, remaining_marked_frames


def propagate_masks(neighborhood, worst_frame_idx, neig_start_idx, ofs):
    marked_frame_idx = -1
    gt_frame_idx = worst_frame_idx - neig_start_idx

    read_pred = lambda x: x
    write_pred = lambda x,y,z: None
    tmp = utils.segmentation.single2multichannel(neighborhood)
    forward_prop = pi.propagate_predictions_memory_conservative(tmp[gt_frame_idx:], multichannel=True,
                                                                of_fns=ofs[gt_frame_idx:],
                                                                orig_fns=tmp[gt_frame_idx:],
                                                                read_frame=read_pred, read_of=read_pred,
                                                                write_frame=write_pred,
                                                                out_dir='./nothing_should_be_here',
                                                                return_preds=True)[1]
    forward_prop = utils.segmentation.multi2singlechannel(forward_prop)
    # backward_prop = pi.propagate_predictions_memory_conservative(tmp[:gt_frame_idx+1][::-1], multichannel=True,
    #                                                              of_fns=ofs[:gt_frame_idx+1][::-1],
    #                                                              orig_fns=tmp[:gt_frame_idx+1][::-1],
    #                                                              read_pred=read_pred, read_of=read_pred,
    #                                                              write_pred=write_pred,
    #                                                              out_dir='./nothing_should_be_here',
    #                                                              return_preds=True,
    #                                                              reverse=True)[1]
    backward_prop = pi.propagate_predictions_memory_conservative(tmp[:gt_frame_idx+1], multichannel=True,
                                                                 of_fns=ofs[:gt_frame_idx+1],
                                                                 orig_fns=tmp[:gt_frame_idx+1],
                                                                 read_frame=read_pred, read_of=read_pred,
                                                                 write_frame=write_pred,
                                                                 out_dir='./nothing_should_be_here',
                                                                 return_preds=True,
                                                                 reverse=True, initial_frame_idx=gt_frame_idx)[1]
    backward_prop = utils.segmentation.multi2singlechannel(backward_prop)
    backward_prop = backward_prop[::-1]

    print(gt_frame_idx, np.unique(neighborhood[gt_frame_idx], return_counts=True), neighborhood.shape, tmp.shape,
          np.unique(tmp[gt_frame_idx], return_counts=True),
          np.unique(forward_prop[0], return_counts=True),
          np.unique(backward_prop[-1], return_counts=True))

    return backward_prop, forward_prop, marked_frame_idx


def propagate_masks_old(neighborhood, worst_frame_idx, neig_start_idx, ofs):
    # TODO: is it necessary to do forward and backward propagation separately? propagate_prediction already supports both or instead, only from the gt propagate forward and backward

    marked_frame_idx = -1

    # forward_prop = pi.propagate_predictions(neighborhood.copy())[0]
    # backward_prop = pi.propagate_predictions(neighborhood[::-1].copy())[0]

    read_pred = lambda x: x
    write_pred = lambda x,y,z: None
    tmp = utils.segmentation.single2multichannel(neighborhood)
    forward_prop = pi.propagate_predictions_memory_conservative(tmp, multichannel=True, of_fns=ofs,
                                                                orig_fns=tmp,
                                                                read_frame=read_pred, read_of=read_pred,
                                                                write_frame=write_pred,
                                                                out_dir='./nothing_should_be_here',
                                                                return_preds=True)[1]
    forward_prop = utils.segmentation.multi2singlechannel(forward_prop)
    # backward_prop = pi.propagate_predictions_memory_conservative(tmp[::-1], multichannel=True, of_fns=ofs[::-1],
    #                                                              orig_fns=tmp[::-1],
    #                                                              read_pred=read_pred, read_of=read_pred,
    #                                                              write_pred=write_pred,
    #                                                              out_dir='./nothing_should_be_here',
    #                                                              return_preds=True,
    #                                                              reverse=True)[1]

    backward_prop = pi.propagate_predictions_memory_conservative(tmp, multichannel=True, of_fns=ofs,
                                                                 orig_fns=tmp,
                                                                 read_frame=read_pred, read_of=read_pred,
                                                                 write_frame=write_pred,
                                                                 out_dir='./nothing_should_be_here',
                                                                 return_preds=True,
                                                                 reverse=True, initial_frame_idx=len(tmp) - 1)[1]
    backward_prop = utils.segmentation.multi2singlechannel(backward_prop)

    backward_prop = backward_prop[::-1]

    gt_frame_idx = worst_frame_idx - neig_start_idx
    print(gt_frame_idx, sorted(np.unique(forward_prop[gt_frame_idx], return_counts=True)[1]), sorted(np.unique(backward_prop[gt_frame_idx], return_counts=True)[1]))
    # assert sorted(np.unique(forward_prop[gt_frame_idx], return_counts=True)[1]) == sorted(np.unique(backward_prop[gt_frame_idx], return_counts=True)[1])

    # If the labels doesn't match, use only the more reliable (non-zero/longer) propation
    if not np.all(forward_prop[gt_frame_idx] == backward_prop[gt_frame_idx]):
        print('more research is needed', gt_frame_idx)
        if np.all(neighborhood[0] == 0):
            print('first 0')
            forward_prop = backward_prop
            marked_frame_idx = neig_start_idx
        elif np.all(neighborhood[-1] == 0):
            print('last 0')
            backward_prop = forward_prop
            marked_frame_idx = neig_start_idx + len(neighborhood) - 1
        else:
            print('other')
            if gt_frame_idx > len(neighborhood) - gt_frame_idx:
                forward_prop = backward_prop
                marked_frame_idx = neig_start_idx
            else:
                backward_prop = forward_prop
                marked_frame_idx = neig_start_idx + len(neighborhood) - 1
    return backward_prop, forward_prop, marked_frame_idx


def select_neighboring_predictions(confidence_scores, marked_frames, nearest_frames, masks, ofs, frame_idx=None):
    if frame_idx is None:
        worst_marked_idx = np.argmin(confidence_scores)
    else:
        worst_marked_idx = np.argmin(np.abs(np.asarray(marked_frames) - frame_idx))  # nearest from marked_frames, then find neighborhood

    worst_frame_idx = marked_frames[worst_marked_idx]
    prev_annot_idx, next_annot_idx = nearest_frames[worst_marked_idx]
    print(confidence_scores)
    print(sum(confidence_scores), confidence_scores[worst_marked_idx], worst_frame_idx, prev_annot_idx, next_annot_idx)

    neig_start_idx, neig_end_idx = max(0, prev_annot_idx), max((next_annot_idx == -1)*len(masks), next_annot_idx + 1)
    neighborhood = np.zeros_like(masks[neig_start_idx:neig_end_idx])
    neighborhood[0] = masks[prev_annot_idx] if prev_annot_idx != -1 else 0
    neighborhood[-1] = masks[next_annot_idx] if next_annot_idx != -1 else 0
    return neighborhood, worst_frame_idx, neig_start_idx, neig_end_idx, worst_marked_idx, ofs[neig_start_idx:neig_end_idx]


def load_annotations(mask_base_dir, gt_base_dir, of_dir_base, seq, read_scores=True):
    mask_dir = pi.get_directory(mask_base_dir, seq)
    #masks = utils.io.read_arrays(utils.io.list_directory(mask_dir + '_instances', sort_key=utils.io.filename_to_number), utils.io.read)
    masks = utils.io.read_arrays(utils.io.list_directory(mask_dir, sort_key=utils.io.filename_to_number, extension='.png'))[:-1]
    # TODO: [:-1] because the forward of doesn't have prediction for the last element: instead save a -backward prop?

    # masks = utils.segmentation.multi2singlechannel(masks)
    marked_frames = utils.io.read(mask_dir + '.json')
    if 0 < len(marked_frames) and marked_frames[-1] >= len(masks):
        marked_frames = marked_frames[:-1]

    gt_dir = pi.get_directory(gt_base_dir, seq)
    gts = utils.io.read_arrays(utils.io.list_directory(gt_dir, sort_key=utils.io.filename_to_number)[:-1],
                               utils.io.read) // 127

    of_dir = pi.get_directory(of_dir_base, seq)
    shape = (*masks.shape, 2) if not utils.segmentation.is_multichannel(masks) else (*masks.shape[:-1], 2)
    ofs = utils.io.read_arrays(utils.io.list_directory(of_dir, sort_key=utils.io.filename_to_number),
                               shape=shape, use_cropping=True, use_mmcv=True)

    scores = [[0]]*len(masks)
    if read_scores:
        scores = utils.io.read_files(utils.io.list_directory(mask_dir, sort_key=utils.io.filename_to_number,
                                                             extension='.pkl'))[:-1]
    return marked_frames, masks, gts, ofs, scores


def calculate_confidence_scores(marked_frames, masks, scores, params=None):
    confidence_scores, nearest_frames = [], []
    for current_frame_id in marked_frames:
        confidence, nearest_preceding_frame, nearest_subsequent_frame = estimate_confidence(current_frame_id, masks,
                                                                                            set(marked_frames), scores,
                                                                                            params=params)
        confidence_scores.append(confidence)
        nearest_frames.append((nearest_preceding_frame, nearest_subsequent_frame))
    return confidence_scores, nearest_frames


@dataclass
class ConfidenceParams:
    use_iou: bool = True
    use_average: bool = True
    use_convexity: bool = True
    use_overlap: bool = True
    use_orig_cropped: bool = True


def estimate_confidence(current_frame_id, masks, marked_set, scores, params=None):
    if params is None:
        params = ConfidenceParams(
            use_iou='iou' in exp_name,
            use_average='A' in exp_name,
            use_convexity='C' in exp_name,
            use_overlap='O' in exp_name,
            use_orig_cropped='D' in exp_name,
        )
        # print(f"params: {params}")
    nearest_frame_id, nearest_preceding_frame, nearest_subsequent_frame = find_nearest_reliable_annotation(current_frame_id, len(masks), marked_set)

    temporal_distance = abs(nearest_frame_id - current_frame_id)
    temporal_distance = 1 - temporal_distance/10
    temporal_distance = max(0.1 * temporal_distance, temporal_distance)

    iou = 0
    if params.use_iou:
        (iou,), _ = utils.segmentation.mask_intersection_over_union(masks[current_frame_id] > 0, masks[nearest_frame_id] > 0, use_iou=True)

    score = params.use_average*scores[current_frame_id][0] + params.use_convexity*scores[current_frame_id][1] + \
            params.use_overlap*scores[current_frame_id][2] + params.use_orig_cropped*scores[current_frame_id][3]
    if 0 < score:
        score /= params.use_average + params.use_convexity + params.use_overlap + params.use_orig_cropped
    if math.isnan(score):
        score = 0

    n_values = 1 + params.use_iou + (params.use_average or params.use_convexity or
                                     params.use_overlap or params.use_orig_cropped)

    confidence_score = (temporal_distance + iou + score) / n_values

    return confidence_score, nearest_preceding_frame, nearest_subsequent_frame


# SCORES: [A] deviation_from_avg, [C] convexity, [O] (1 -) overlap, [D] orig-cropped overlap
"""
Confidence scores: see 
!!  original:               "ts": 2, "iou": 0.6780489582289353
OK  original:               "ts": 2, "iou": 0.6775749843957893
OK  old temp + iou:         see overleaf
!!  temp + iou (prop bug):  "ts": 0, "iou": 0.7216051744015837, "confidence_threshold": 0.2, "n_annotated_frames": 18, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_matched_conf_thres0.2_2021117_03_propagate_only_from_the_gt"
NO  temp + iou:             "ts": 1, "iou": 0.7571116398584619, "confidence_threshold": 0.4, "n_annotated_frames": 62, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.4_2021122_07_temp_iou"
NO  temp:                   "ts": 0, "iou": 0.753632906009043,  "confidence_threshold": 0.0, "n_annotated_frames": 57, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.0_2021122_08_temp"
    ==> only scores
NO  temp + (D):             "ts": 0, "iou": 0.753632906009043,  "confidence_threshold": 0.0, "n_annotated_frames": 57, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.0_2021122_06_temp_D"
NO  temp + (AD):            "ts": 0, "iou": 0.7391048257261484, "confidence_threshold": 0.0, "n_annotated_frames": 42, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.0_2021122_06_temp_AD"
OK  temp + (ACD):           "ts": 0, "iou": 0.7414963734070491, "confidence_threshold": 0.0, "n_annotated_frames": 40, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.0_2021122_06_temp_ACD"
NO  temp + (ACOD)           "ts": 0, "iou": 0.7414560827399115, "confidence_threshold": 0.0, "n_annotated_frames": 40,  "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.0_2021122_06_temp_ACOD"
??  temp + (CD):            "ts": 0, "iou": 0.7452619377357221, "confidence_threshold": 0.0, "n_annotated_frames": 48, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.0_2021122_06_temp_CD"
    ==> see whether the iou is needed or not, leave out if not
NO  temp + iou + (ACOD):    "ts": 1, "iou": 0.7415920488752894, "confidence_threshold": 0.3, "n_annotated_frames": 38, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.3_2021122_04_temp_iou_scores"
NO  temp + iou + (D):       "ts": 1, "iou": 0.7330627781941451, "confidence_threshold": 0.2, "n_annotated_frames": 33, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.2_2021122_05_temp_iou_D"
NO  temp + iou + (AD):      "ts": 1, "iou": 0.7316873690957508, "confidence_threshold": 0.2, "n_annotated_frames": 30, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.2_2021122_05_temp_iou_AD"
NO  temp + iou + (ACD):     "ts": 1, "iou": 0.7514894226947415}, "confidence_threshold": 0.3, "n_annotated_frames": 50, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.3_2021122_05_temp_iou_ACD"
NO  temp + iou + (CD):      "ts": 1, "iou": 0.7515636083147412, "confidence_threshold": 0.3, "n_annotated_frames": 51, "/media/hdd2/lkopi/datasets/rats_pipeline/test_set/15020/images_00012_instonly_model_only_clean/prop_results_overlap_thrs_0.1_correct_matched_conf_thres0.3_2021122_05_temp_iou_CD"

"""


def find_nearest_reliable_annotation(current_frame_id, n_frames, marked_set):
    preceding_frame_ids = list(range(current_frame_id-1, -1, -1))
    subsequent_frame_ids = list(range(current_frame_id + 1, n_frames))

    nearest_preceding_frame = next((x for x in preceding_frame_ids if x not in marked_set), -1)
    nearest_subsequent_frame = next((x for x in subsequent_frame_ids if x not in marked_set), -1)

    if nearest_preceding_frame == -1:
        nearest_frame_id = nearest_subsequent_frame
    elif nearest_subsequent_frame == -1:
        nearest_frame_id = nearest_preceding_frame
    else:
        nearest_frame_id = nearest_preceding_frame \
            if abs(nearest_preceding_frame - current_frame_id) <= abs(nearest_subsequent_frame - current_frame_id) \
            else nearest_subsequent_frame

    return nearest_frame_id, nearest_preceding_frame, nearest_subsequent_frame


'''
def find_nearest_reliable_annotation(current_frame_id, masks, marked_set):
    traversal_list = generate_frame_traversal_list(current_frame_id, masks)

    nearest_frame_id = -1
    for fid in traversal_list:
        if fid not in marked_set:
            nearest_frame_id = fid
            break
    return nearest_frame_id


def generate_frame_traversal_list(current_frame_id, masks):
    # TODO: could be imporved using only numpy, e.g. sort arange based on distance
    neighborhood_radius = min(current_frame_id, len(masks) - current_frame_id - 1)

    preceding_frame_ids = list(range(current_frame_id-1, -1, -1))
    subsequent_frame_ids = list(range(current_frame_id+1, len(masks)))
    assert len(preceding_frame_ids[neighborhood_radius:]) == 0 or len(subsequent_frame_ids[neighborhood_radius:]) == 0

    traversal_list = []
    if neighborhood_radius != 0:
        traversal_list = utils.coord_lists_to_contour_list(preceding_frame_ids[:neighborhood_radius],
                                                          subsequent_frame_ids[:neighborhood_radius])
    traversal_list += preceding_frame_ids[neighborhood_radius:] + subsequent_frame_ids[neighborhood_radius:]
    assert len(traversal_list) == len(masks) - 1

    return traversal_list
'''


if __name__ == '__main__':
    run()
