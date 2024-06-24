import os

import numpy as np
from tqdm import tqdm

import utils.io
import utils.segmentation
import utils.visualization
import utils.keypoints
import behavior_detection_rule_based as behavior
import switch_up_behaviors as sub


def switch_up_colors(mask):
    orig_mask = mask.copy()
    mask = utils.segmentation.multi2singlechannel(mask)
    inst_ids = np.unique(mask)[1:]
    if len(inst_ids) == 1:
        inst_id, = inst_ids
        new_id = inst_id + (-1 if 1 < inst_id else 1)
        mask[mask == inst_id] = new_id
    elif len(inst_ids) == 2:
        new_ids = inst_ids[::-1]
        mask = utils.visualization.map_labels(mask, dict(zip(inst_ids, new_ids)))[..., 0]
    elif len(inst_ids) > 2:
        assert False
    tmp = utils.segmentation.single2multichannel(mask)
    orig_mask[..., :tmp.shape[-1]] = tmp
    return orig_mask


def run(behavior_arch, prop_dir, config_file="./config.arch"):
    behaviors = behavior.from_arch(behavior_arch, contain_markers=True)
    # print(behaviors.shape, behaviors[:2])

    switch_locations = np.where(behaviors[:, 1] == 'switch up')[0]
    for l in switch_locations:
        behaviors[l:, :] = behaviors[l:, ::-1]
        behaviors[l, 1] = ''
    behaviors[:, 1][switch_locations] = ''

    with open(behavior_arch, mode='w') as file:
        file.write(behavior.to_arch(behaviors, config_file, contain_markers=True))

    all_fns = []
    for seq in utils.io.list_directory(prop_dir, only_dirs=True, full_path=False):
        fns = utils.io.list_directory(os.path.join(prop_dir, seq), extension='.png',
                                      sort_key=utils.io.filename_to_number)
        all_fns += fns

    switch_locations[1::2] += 1
    switch_locations *= 6
    print(switch_locations)
    switch_intervals = zip(switch_locations[::2], switch_locations[1::2])

    for switch_start, switch_end in switch_intervals:
        print(switch_start, switch_end)
        utils.io.modify_parallel(all_fns[switch_start:switch_end], switch_up_colors)
        sub.run(behavior_arch, switch_start, keep_orig=False, contain_markers=True, use_frame_id=True)
        sub.run(behavior_arch, switch_end, keep_orig=False, contain_markers=True, use_frame_id=True)

    if len(switch_locations) % 2 != 0:
        print(switch_locations[-1])
        switch_start = switch_locations[-1]

        utils.io.modify_parallel(all_fns[switch_start:], switch_up_colors)
        sub.run(behavior_arch, switch_start, keep_orig=False, contain_markers=True, use_frame_id=True)


if __name__ == "__main__":
    behavior_arch = "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_1_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1_behaviors_conv0.0/behaviors_dt_switched_thrs0.3_suggested111_fixed.arch"
    prop_dir = "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_1_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1_conv0.0"

    run(behavior_arch, prop_dir)
