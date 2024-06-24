import os
import sys
sys.path.append('../')

import numpy as np
from tqdm import tqdm

import utils.io
import utils.segmentation
import utils.visualization
import propagate_instances
import iterative_training
import visualization
import behavior_detection_rule_based as behavior
import switch_up_behaviors
import position_statistics

# img_dir = "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_3"
# img_dir = "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_1"
# img_dir = "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA animals/1-Autism study_autistic animals_3"
# of_dir_format = img_dir + "/{}/GMA"
img_dirs = [
    #"/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_1",
    #"/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA animals/1-Autism study_autistic animals_3",
    #"/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_3",
    "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA animals/1-Autism study_autistic animals_1",
]

mask_dir_formats = [
    # "/media/hdd2/lkopi/datasets/anna/20220224/ptsNedges_v2_aug_mod1tr",  # control3
    # "/media/hdd2/lkopi/datasets/anna/20220302/ptsNedges_v1_nblre4_rs",  # control1
    # "/media/hdd2/lkopi/datasets/anna/20220310/result_segm_v4_col3r_ep14tr",  # animals3
    # "/media/hdd2/lkopi/datasets/anna/20220312/control1",
    # "/media/hdd2/lkopi/datasets/anna/20220312/animals3",
    #"/media/hdd2/lkopi/datasets/anna/20220321/control1",
    #"/media/hdd2/lkopi/datasets/anna/20220316/animals3",
    # "/media/hdd2/lkopi/datasets/anna/20220324/control3",
    "/media/hdd2/lkopi/datasets/anna/20220402/animals1",
]

all_params = [
    # {"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0., "max_deviation":0.5},  # 3  OK
    # {"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0.},  # conv0.0 -- control 1: 2 TS
    # {"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0.5},  # conv0.5 -- same, nothing changed
    # {"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0.6},  # conv0.6 -- maybe?
    # CONTROL1: it fixes 09:05, but 30:25 needs stricter rules
    # ANIMALS3: 8:58.80 and 20:53.00 + dropped frames when grooming ~2-3 min
    # {"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0.6, "ksize":3},  # removed -- nope, drops almost everything
    # {"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0.6, "max_deviation":0.3},  # maybe?
    # {"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0.6, "max_deviation":0.2},  # maybe?
    # {"overlap_threshold" :0.1, "return_original" :True, "convexity_threshold" :0.6, "remove_disproportionate_instances" :False},  # conv0.6_nodisprop -- maybe?
    # CONTROL1: ~6:00 TS due to merged segments
    {"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0.6, "max_deviation":0.6},  # param0
    {"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0.4, "max_deviation":0.6},  # param1
    #{"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0.6, "max_deviation":0.5},  # param2
    #{"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0.5, "max_deviation":0.5},  # param3
    #{"overlap_threshold":0.1, "return_original":True, "convexity_threshold":0.4, "max_deviation":0.5},  # param4
]
RUN_ALL = True
RUN_BEHAVIORS = False
n_finished = 1

seqs = utils.io.list_directory(img_dirs[0], only_dirs=True, full_path=False)  # [:-1]
# seqs = ['seq004', 'seq010']
# seqs = ['seq015']
for param_idx, params in enumerate(all_params):
    if param_idx < n_finished:
        continue
    print(param_idx)
    for img_dir, mask_dir_format in zip(img_dirs, mask_dir_formats):
        of_dir_format = img_dir + "/{}/GMA"

        out_dir = f"{mask_dir_format}_param{param_idx}_prop_results_nonconvex"

        if not os.path.exists(out_dir):
            utils.io.save(os.path.join(out_dir, f"params{param_idx}.json"), params)
            propagate_instances.propagate_sequences_memory_conservative(seqs, mask_dir_format, out_dir,
                                                                        of_dir_base=of_dir_format,
                                                                        params=params, chunksize=200)

        if not RUN_ALL:
            if not os.path.exists(f"{out_dir}_vis.mp4"):
                visualization.visualize_predictions_memory_conservative(img_dir, out_dir, seqs=seqs,
                                                                        concat_videos=True,
                                                                        n_processes=len(seqs),
                                                                        out_fn=f"{out_dir}_vis.mp4")
            else:
                print(f"[SKIP] {out_dir}_vis.mp4")
        else:
            if not RUN_BEHAVIORS:
                seqs_to_switch_up = propagate_instances.match_label_ids_between_sequences(out_dir)
                utils.io.save(os.path.join(out_dir + '_behaviors', 'switched_sequences.json'), seqs_to_switch_up)
            else:
                behavior.run(instances_dir=out_dir,
                             base_arch_file="/home/lkopi/repo/rat_segmentation_public/config.arch",
                             output_dir=out_dir + '_behaviors',
                             eval=False, merge=True, verbose=3)

                switch_up_behaviors.match_behaviors_between_sequences(out_dir, out_dir + '_behaviors')

            position_statistics.run(out_dir, switched_sequences=out_dir + '_behaviors/switched_sequences.json',
                                    inner_box_xyxy=[218, 16, 442, 406], inner_box_size_in_meters=[0.4, 0.8],
                                    n_processes=16, chunksize=200,
                                    img_dir_base=img_dir)

            visualization.visualize_predictions_memory_conservative(img_dir, out_dir,
                                                                    switched_sequences=out_dir + '_behaviors/switched_sequences.json',
                                                                    concat_videos=True,
                                                                    n_processes=16,
                                                                    out_fn=f"{out_dir}_vis.mp4")
