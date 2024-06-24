import datetime
import os

import numpy as np

import behavior_detection_rule_based as behavior
import propagate_instances
import utils.io


def remove_rat_index(arr):
    shape = arr.shape
    return np.array(list(map(lambda behavior: behavior[:-2], arr.flatten()))).reshape(shape)


def to_arch(behaviors, base_arch_file):
    import re

    with open(base_arch_file, mode='r') as file:
        arch_file = file.read()

    behavior_list = re.findall(pattern='CBName\t(.+)\n', string=arch_file)
    rep_str = ''.join([f'{behavior_list.index(b1) + 1}\t\t{behavior_list.index(b2) + 1}\n' for (b1, b2) in behaviors])
    return re.sub(pattern='(?s)(0.000\t).*', repl=f'\g<1>{rep_str}', string=arch_file)


def from_arch(arch_file):
    import re

    with open(arch_file, mode='r') as file:
        arch_file = file.read()

    behavior_list = re.findall(pattern='CBName\t(.+)\n', string=arch_file)
    behaviors = re.search(pattern='(?s)0.000\t(.*)', string=arch_file).group(1)
    return np.array(behavior_list)[np.array(behaviors.split(), dtype=int).reshape(-1, 2) - 1]


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--arch_file', required=True, help='Coding sheet.')
    parser.add_argument('-t', '--t_switch', required=True, help='Instant of time at which switching behaviors occurs.')
    return parser.parse_args()


def run(arch_file, t_switch, keep_orig=True, contain_markers=False, use_frame_id=False):
    # python switch_up_behaviors.py -c behaviors_dt.arch -t 00:00:1.20
    DT = .2

    behaviors = behavior.from_arch(arch_file=arch_file, contain_markers=contain_markers)

    if use_frame_id:
        frame_id = t_switch
    else:
        t = datetime.datetime.strptime(t_switch,'%H:%M:%S.%f')
        t_sec = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond).total_seconds()
        frame_id = round(t_sec / DT)

    behaviors = np.vstack([behaviors[:frame_id], behaviors[frame_id:][..., ::-1]])
    if contain_markers:  # remove rat indexes
        behaviors = np.array([[e[0][:-2] + ' 1', e[1], e[2][:-2] + ' 2'] for e in behaviors])
    else:
        behaviors = np.array([[e[0][:-2] + ' 1', e[1][:-2] + ' 2'] for e in behaviors])
    arch_out = os.path.join(os.path.dirname(arch_file),
                            os.path.splitext(os.path.basename(arch_file))[0] + ('_switched.arch' if keep_orig else '.arch'))
    data = behavior.to_arch(behaviors=behaviors, base_arch_file=arch_file, contain_markers=contain_markers)
    with open(arch_out, mode='w') as file:
        file.write(data)


def match_behaviors_between_sequences(pred_dir, behavior_dir):
    if not os.path.exists(os.path.join(behavior_dir, 'behaviors_dt_backup')):
        utils.io.copy_directory(os.path.join(behavior_dir, 'behaviors_dt'),
                                os.path.join(behavior_dir, 'behaviors_dt_backup'))
    else:
        utils.io.copy_directory(os.path.join(behavior_dir, 'behaviors_dt_backup'),
                                os.path.join(behavior_dir, 'behaviors_dt'))
    seqs_to_switch_up = propagate_instances.match_label_ids_between_sequences(pred_dir)[:-1]
    for seq in seqs_to_switch_up:
        print(seq)
        run(os.path.join(behavior_dir, 'behaviors_dt', seq + '.arch'), '00:00:0.00', keep_orig=False)
    with open(os.path.join(behavior_dir, 'behaviors_dt_switched.arch'), mode='w') as file:
        file.write(behavior.merge_arch(arch_dir=os.path.join(behavior_dir, 'behaviors_dt')))
    utils.io.save(os.path.join(behavior_dir, 'switched_sequences.json'), seqs_to_switch_up)


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
