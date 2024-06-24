import math
import os

import numpy as np

import utils.io
import utils.segmentation
import utils.keypoints
import propagate_instances
import eval_propagation
import annotation
import behavior_detection_rule_based as behavior


def run(prop_dir, behavior_arch, out_file=None, config_file='config_new.arch', params=None, conf_thres=0.3):
    if params is None:
        params = annotation.ConfidenceParams(
            use_iou=False,
            use_average=True,  # A
            use_convexity=True,  # C
            use_overlap=False,  # O
            use_orig_cropped=True,  # D
        )

    behaviors = behavior.from_arch(behavior_arch)
    every_missed_frame = []

    seqs = utils.io.list_directory(prop_dir, only_dirs=True, full_path=False)
    for seq in seqs:
        missed_json = os.path.join(prop_dir, seq + ".json")
        dropped_frames = utils.io.read(missed_json)
        # print(missed_frames)

        ########################################
        ### Mark missed seqs and end of seqs ###
        ########################################
        seq_length = len(utils.io.list_directory(os.path.join(prop_dir, seq), extension='.png'))
        tmp = np.zeros((seq_length + 5) // 6 * 6, dtype=np.uint8)
        tmp[dropped_frames] = 1

        missed_count = np.sum(tmp.reshape(-1, 6), axis=-1)
        missed_start = missed_count > 3
        missed_subseq = missed_count == 6
        # Remove length of 1 sequences:
        # missed_subseq[1:-1][np.logical_and(~missed_subseq[:-2], ~missed_subseq[2:])] = False # Removes 13:01.20...
        # Extend it with subseqent unsure timestep
        missed_almost = missed_count > 3
        missed_subseq[1:][np.logical_and(missed_subseq[:-1], missed_almost[1:])] = True
        # missed_end = missed_count < 6  # <= 3
        # Remove inbetween checks
        missed_subseq[1:-1][np.logical_and(missed_subseq[:-2], missed_subseq[2:])] = True

        missed_frames = np.zeros_like(missed_count, dtype=np.uint8)
        missed_frames[missed_subseq] = 1
        missed_frames[1:][np.logical_and(missed_subseq[:-1], ~missed_subseq[1:])] = 2

        if missed_frames[-1] == 1:
            missed_frames[-1] = 2

        ########################################
        ###    Calculate confidence scores   ###
        ########################################
        if len(dropped_frames) == 0:
            every_missed_frame += missed_frames.tolist()
            continue
        ### 1. Find start and end of each missed sequences
        start_frames, end_frames, lengths = propagate_instances.select_subsequent_frame_ids(dropped_frames)
        # print(start_frames, end_frames, lengths)
        start_frames_div6 = start_frames // 6
        end_frames_div6 = end_frames // 6  # (end_frames + 5)  // 6
        # if start_frames_div6[-1] + 1 >= len(missed_frames):
        #    start_frames_div6[-1] = len(missed_frames) - 2
        start_frames_div6 = np.clip(start_frames_div6, 0, len(missed_frames) - 2)
        if end_frames_div6[-1] == len(missed_frames):
            end_frames_div6[-1] -= 1
        probable = np.logical_and(np.logical_and(
            np.logical_or(missed_frames[start_frames_div6] == 1, missed_frames[start_frames_div6 + 1] == 1),
            np.logical_or(missed_frames[end_frames_div6] == 1, missed_frames[end_frames_div6] == 2)),
            lengths > 6)
        start_frames, end_frames, lengths = start_frames[probable], end_frames[probable], lengths[probable]
        # print(start_frames, end_frames, lengths, np.where(missed_frames == 2))

        ### 2. Calculate scores: read conf scores and calc iou between the first and last (next after) mask
        conf_scores = np.zeros_like(start_frames, dtype=float)
        conf_score_file_format = os.path.join(prop_dir, seq, "{:05d}.pkl")
        mask_file_format = conf_score_file_format.replace('.pkl', '.png')
        for seq_idx, (start_frame, end_frame, length) in enumerate(zip(start_frames, end_frames + 1, lengths)):
            conf_score = 0
            for frame_idx in range(start_frame, end_frame):
                scores = utils.io.read(conf_score_file_format.format(frame_idx))

                # score = params.use_average * scores[0] + params.use_convexity * scores[1] + \
                #         params.use_overlap * scores[2] + params.use_orig_cropped * scores[3]
                try:
                    score = params.use_average * scores[0] + \
                            params.use_overlap * scores[1] + params.use_orig_cropped * scores[2]
                except:
                    score = 0

                if 0 < score:
                    score /= params.use_average + params.use_convexity + params.use_overlap + params.use_orig_cropped
                if math.isnan(score):
                    score = 0
                conf_score += score
            conf_score /= length

            mask1 = utils.segmentation.multi2singlechannel(utils.io.read(mask_file_format.format(start_frame)))
            try:
                mask2 = utils.segmentation.multi2singlechannel(utils.io.read(mask_file_format.format(end_frame)))
            except:
                mask2 = utils.segmentation.multi2singlechannel(utils.io.read(mask_file_format.format(end_frame - 1)))
            # print(np.unique(mask1), np.unique(mask2))
            try:
                iou = eval_propagation.calulate_average_iou(mask1, mask2)
            except:
                iou = 0
            if iou == 1:
                iou = 0
            conf_score += iou
            conf_score /= 2
            conf_scores[seq_idx] = conf_score

        print(seq, len(conf_scores), np.count_nonzero(conf_scores < conf_thres), start_frames, end_frames, conf_scores,
              np.where(missed_frames == 2))

        ### 3. Find nearest 'check' and remove it if the avg score < thrs
        bad_options = (end_frames[conf_scores < conf_thres] // 6).tolist()
        all_options = set(np.where(missed_frames == 2)[0].tolist())
        needs_checking = set(
            map(lambda f: next((x for x in range(f, len(missed_frames)) if x in all_options), -1), bad_options))
        # print(len(bad_options), bad_options, all_options, needs_checking)
        missed_frames[np.asarray(list(needs_checking), dtype=np.int32)] = 3

        ### 4. Check whether all seqs is found, and print stats (all/found/suggested)

        every_missed_frame += missed_frames.tolist()
    if len(every_missed_frame) < len(behaviors):
        every_missed_frame.append(1)
    every_missed_frame = np.asarray(every_missed_frame)
    markers = np.zeros(len(behaviors), dtype='<U50')
    markers[every_missed_frame == 1] = 'missed'
    markers[every_missed_frame == 2] = 'missed'
    markers[every_missed_frame == 3] = 'check'
    n_suggested = np.count_nonzero(every_missed_frame == 3)
    # print(markers)
    behaviors_with_markers = np.zeros((len(behaviors), 3), dtype='<U50')
    behaviors_with_markers[:, 0] = behaviors[:, 0]
    behaviors_with_markers[:, -1] = behaviors[:, -1]
    behaviors_with_markers[:, 1] = markers
    if out_file is None:
        out_file = f"{os.path.splitext(behavior_arch)[0]}_thrs{conf_thres}_suggested{n_suggested}.arch"
    with open(out_file, mode='w') as file:
        file.write(behavior.to_arch(behaviors_with_markers, config_file, contain_markers=True))


if __name__ == '__main__':
    # prop_dir = "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA animals/1-Autism study_autistic animals_3_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1_conv0.5"
    # behavior_arch = "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA animals/1-Autism study_autistic animals_3_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1_behaviors_conv0.5/behaviors_dt_switched.arch"

    #prop_dir = "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_1_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1"
    '''
    prop_dirs = [
        "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_3_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1",
        "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA animals/1-Autism study_autistic animals_1_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1",
    ]
    '''
    prop_dirs = [
        # "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_3_model_0079999.pth/prop_results_overlap_thrs_0.1",
        # "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA animals/1-Autism study_autistic animals_3_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1"
        "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA animals/1-Autism study_autistic animals_2_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1",
        "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_2_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1"
    ]

    for prop_dir in prop_dirs:
        behavior_arch = f"{prop_dir}_behaviors/behaviors_dt_switched.arch"
        run(prop_dir, behavior_arch)
