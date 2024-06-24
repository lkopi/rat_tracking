import os

import numpy as np

import behavior_detection_rule_based as behavior
import switch_up_behaviors
import utils.io


config_file = "config_new.arch"
# behaviors_gt_dir = "/media/hdd2/lkopi/datasets/rats_clean/test_videos/behavior"
behaviors_gt_dir = "/home/ahmad/rats_behavior/behavior_annotations"
instances_dir = "/media/hdd2/lkopi/datasets/anna/2021_11_30/ptsNedges_v2_nblre4_rs_param7_prop_results_nonconvex_matched"

labels = [
        'chasing', 'head-to-head contact', 'body sniffing', 'side-to-side contact',
        'mounting', 'being mounted', 'non-social behaviour',
]

"""
ALIGN_BOT_TH, ALIGN_H_TH = 12, 15
CLOSE_TH_1, CLOSE_TH_2 = .15, .09
TOUCH_TH = .02
CHASING_TH = .5
MOUNTING_TH = 1400  # TODO make it smaller  # 1700
SIDE_L_TH_1, SIDE_L_TH_2, SIDE_D_TH, PASSIVE_TH_1, PASSIVE_TH_2 = .35, .5, .1, .07, .2
PARALLEL_TH = 60
HEAD_D_TH = .1    # TODO make it larger  # .075
EPS = .009
"""

side0 = (.35, .5, .1)
side1 = (.5, .75, .125)
side2 = (.75, .1, .15)
side3 = (.25, .75, .125)
clos0 = (.15, .09)
clos1 = (.75, .045)
clos2 = (.075, .045)
#                    Nope  Worse Better   Same  Worse  Worse Better ...
#                       0      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15     16
MOUNTING_TH_list = [ 1700,  1400,  2000,  1700,  2400,  2400,  2000,  2000,  2000,  2000,  2400,  1700,  2000,  2000,  2000,  2000,  2000,  2000,  2000]
HEAD_D_TH_list   = [ .075,    .1,    .1,    .1,    .1,    .2,    .1,    .1,    .1,    .2,    .2,    .2,    .2,    .2,   .05,   .05,   .05,    .1,    .1]
SIDE_params_list = [side0, side0, side0, side0, side0, side0, side1, side1, side2, side2, side2, side2, side2, side2, side1, side1, side3, side1, side1]
CLOS_params_list = [clos0, clos0, clos0, clos0, clos0, clos0, clos0, clos1, clos1, clos1, clos1, clos1, clos1, clos1, clos0, clos2, clos2, clos0, clos0]
CHASING_TH_list  = [   .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,    .3,    .8,    .5,    .5,    .5,   .25,   .75]

for param_id in range(18, 19): #len(MOUNTING_TH_list)):
    behavior.MOUNTING_TH = MOUNTING_TH_list[param_id]
    behavior.HEAD_D_TH = HEAD_D_TH_list[param_id]
    behavior.SIDE_L_TH_1, behavior.SIDE_L_TH_2, behavior.SIDE_D_TH = SIDE_params_list[param_id]
    behavior.CLOSE_TH_1, behavior.CLOSE_TH_2 = CLOS_params_list[param_id]
    behavior.CHASING_TH = CHASING_TH_list[param_id]
    output_dir = f"{instances_dir}_behavior_param{param_id}"
    print(output_dir, behavior.MOUNTING_TH, behavior.HEAD_D_TH, SIDE_params_list[param_id], CLOS_params_list[param_id])
    #behavior.run(instances_dir=instances_dir, base_arch_file=config_file,
    #             output_dir=output_dir, eval=False, merge=True, verbose=1,
    #             n_processes=16)
    # switch_up_behaviors.match_behaviors_between_sequences(instances_dir, output_dir)
    behavior.evaluate_behaviors(behaviors_gt_dir, os.path.join(output_dir, 'behaviors_dt'), output_dir, labels=labels)


# param14: same
# param15: mixed: old slighly better, new slightly worse
# param16: better
# param18: chasing same

