import numpy as np

import behavior_detection_rule_based as behavior
import switch_up_behaviors
import utils.io


# Prepare GT
behavior_arch_a = "temp/1-Autism study_control animals_1-A.arch"
annots_a = behavior.from_arch(behavior_arch_a)
behavior_arch_b = "temp/1-Autism study_control animals_1-B.arch"
annots_b = behavior.from_arch(behavior_arch_b)

merged = np.stack((annots_a.reshape(-1), annots_b.reshape(-1)), axis=-1)
# merged[merged == 'non-social behaviour'] = 'Non-social behaviour'
merged[merged == 'self grooming'] = 'non-social behaviour'  # 'social grooming'
merged = np.array([[e[1] + ' 1', e[0] + ' 2'] for e in merged])

config_file = "config_new.arch"
utils.io.make_directory('temp/control1_gt')
out_file = "temp/control1_gt/1-Autism study_control animals_1-A-B.arch"
with open(out_file, mode='w') as file:
    file.write(behavior.to_arch(merged, config_file))


behaviors_gt_dir = "temp/control1_gt"
instances_dir = "/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_1_separats/prop_results_overlap_thrs_0.1"

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
clos3 = (.037, .022)
#                    Nope  Worse Better   Same  Worse  Worse Better ...
#                       0      1      2      3      4      5      6      7      8      9     10,    11,    12,    13,    14     15
MOUNTING_TH_list = [ 1700,  1400,  2000,  1700,  2400,  2400,  2000,  2000,  2000,  2000,  2400,  2000,  2000,  2000,  2000,  2000]
HEAD_D_TH_list   = [ .075,    .1,    .1,    .1,    .1,    .2,    .1,    .1,    .1,    .2,    .2,   .05,   .05,   .05,    .1,    .1]
SIDE_params_list = [side0, side0, side0, side0, side0, side0, side1, side1, side2, side2, side2, side1, side1, side3, side1, side1]
CLOS_params_list = [clos0, clos0, clos0, clos0, clos0, clos0, clos0, clos1, clos1, clos1, clos1, clos0, clos2, clos2, clos2, clos3]
CHASING_TH_list  = [   .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,    .5,   .75,   .75]

for param_id in range(15, len(MOUNTING_TH_list)):
    behavior.MOUNTING_TH = MOUNTING_TH_list[param_id]
    behavior.HEAD_D_TH = HEAD_D_TH_list[param_id]
    behavior.SIDE_L_TH_1, behavior.SIDE_L_TH_2, behavior.SIDE_D_TH = SIDE_params_list[param_id]
    behavior.CLOSE_TH_1, behavior.CLOSE_TH_2 = CLOS_params_list[param_id]
    output_dir = f"{instances_dir}_behavior_param{param_id}"
    print(output_dir, behavior.MOUNTING_TH, behavior.HEAD_D_TH, SIDE_params_list[param_id], CLOS_params_list[param_id])
    behavior.run(instances_dir=instances_dir, base_arch_file=config_file,
                 output_dir=output_dir, eval=False, merge=True, verbose=1,  # 3,
                 n_processes=16)
    switch_up_behaviors.match_behaviors_between_sequences(instances_dir, output_dir)
    output_dir2 = f"{output_dir}/eval"
    utils.io.copy_file(f"{output_dir}/behaviors_dt_switched.arch", f"{output_dir2}/behaviors_dt_switched.arch")
    behavior.evaluate_behaviors(behaviors_gt_dir, output_dir2, output_dir2)



"""
param_id = 0
output_dir = f"{instances_dir}_behavior_param{param_id}"
behavior.run(instances_dir=instances_dir, base_arch_file=config_file,
             output_dir=output_dir, eval=False, merge=True, verbose=3,
             n_processes=16)
behavior.evaluate_behaviors(behaviors_gt_dir, output_dir, output_dir)
"""