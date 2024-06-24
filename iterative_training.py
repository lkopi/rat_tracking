#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import multiprocessing
import os
import random

import propagate_instances as pi
import propagate_instances_old as pi_old
import eval_propagation as ep
import utils.io


def propagate_detectron2(model_dir, out_dir, overlap_thrs=0.1, params_json=None, use_old=True, visualize=False,
                         img_base_dir=None, n_processes=6, chunksize=9999, save_only_scores=False):

    # Find sequences
    '''
    seqs = utils.io.list_directory('{}/eval_results'.format(model_dir), only_dirs=True, full_path=False)
    img_dir = model_dir + '/eval_results/{}/im'
    inst_dir = model_dir + '/eval_results/{}/annot'
    '''
    seqs = utils.io.list_directory('{}/result_segm_v4.4_col3ep14'.format(model_dir), only_dirs=True, full_path=False)
    img_dir = model_dir + '/result_segm_v4.4_col3ep14/{}'
    inst_dir = model_dir + '/result_segm_v4.4_col3ep14/{}'
    of_dir = None
    if img_base_dir is not None:
        of_dir = img_base_dir + '/{}/GMA'

    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, len(seqs))

    params = None
    if params_json is not None:
        params = utils.io.read(params_json)

    if use_old:
        pi_old.propagate_sequences(seqs, img_dir, inst_dir, out_dir, overlap_thrs=overlap_thrs, visualize=visualize)
    else:
        for i in range((len(seqs)+n_processes-1) // n_processes):
            pi.propagate_sequences_memory_conservative(seqs[i*n_processes:(i+1)*n_processes],
                                                       inst_dir, out_dir, overlap_thrs=overlap_thrs,
                                                       params=params, of_dir_base=of_dir, chunksize=chunksize,
                                                       save_only_scores=save_only_scores)


def collect_prev_fns(prev_fn):
    fns = []
    annots = utils.io.read(prev_fn)
    for annot in annots:
        fns.append(annot['file_name'])
    return fns


def select_bad_samples(out_fn, prop_dir, eval_dir, gt_annot, n_train, seed=0, prev_fn=None):
    # Collect bad samples
    bad_seqs = utils.io.list_directory(prop_dir, fn_constraint=lambda fn: fn.endswith('.json'))
    all_samples = []
    for bad_seq in bad_seqs:
        bad_frame_ids = utils.io.read(bad_seq)
        seq_name = os.path.splitext(os.path.basename(bad_seq))[0]
        print(seq_name, bad_frame_ids)
        bad_samples = list(map(lambda fn: os.path.join(eval_dir, seq_name, '{}.png'.format(fn)), bad_frame_ids))
        all_samples += bad_samples

    # Keed only those that are in the dataset
    annots = utils.io.read(gt_annot)
    all_samples = set(all_samples)
    prev_fns = set()
    if prev_fn is not None:
        prev_fns = set(collect_prev_fns(prev_fn))
    train_annots = []
    all_train_annots = []
    for annot in annots:
        if annot['file_name'] in all_samples and annot['file_name'] not in prev_fns:
            all_train_annots.append(annot)

    # Sample from the remaining
    if seed is not None:
        random.seed(seed)
    if n_train < len(all_train_annots):
        train_annots = random.sample(all_train_annots, n_train)
    if prev_fn is not None:
        prev_annots = utils.io.read(prev_fn)
        train_annots = prev_annots + train_annots
        all_train_annots = prev_annots + all_train_annots
    print(len(all_samples), len(prev_fns), len(all_train_annots), len(train_annots), out_fn)

    # Save the dataset
    utils.io.save(out_fn, train_annots)
    utils.io.save(out_fn.replace('.json', '_all.json'), all_train_annots)
    return len(train_annots)


def evaluate_sequence_wrapper(params):
    return ep.evaluate_sequence(*params)


def evaluate_results(pred_dir, gt_dir):
    # Initialization
    pool = multiprocessing.Pool()
    n_switches = 0
    seqs = utils.io.list_directory(gt_dir, only_dirs=True, full_path=False)
    if len(seqs) == 0:
        seqs = ['']
        pred_dir = os.path.join(pred_dir, 'images')
    cumulated_iou = 0
    all_frames = 0
    results = {}

    # Evaluate each sequence
    for ns, avg_iou, n_frames, seq in pool.imap_unordered(evaluate_sequence_wrapper,
                                                          zip(seqs,
                                                              itertools.repeat(gt_dir),
                                                              itertools.repeat(pred_dir))):
        results[seq] = (ns, avg_iou)
        n_switches += ns
        print(seq, ns, avg_iou, n_frames)
        cumulated_iou += n_frames * avg_iou
        all_frames += n_frames
    pool.close()
    pool.join()

    cumulated_iou /= all_frames
    print(n_switches, cumulated_iou)
    results['final'] = (n_switches, cumulated_iou)
    utils.io.save(pred_dir + '/eval.json', results)
    return n_switches, cumulated_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', required=True, help='Experiment name.')
    parser.add_argument('-m', '--model_dir', required=True, help='Directory of the pre-trained model.')
    parser.add_argument('-gt', '--gt_dir', required=True, help='Directory of the test set.')
    parser.add_argument('-gta', '--gt_annot', required=True,
                        help='Json file containing the annotations of the test set.')
    parser.add_argument('-p', '--prop_dir', required=True, help='Directory to save the propagated results.')
    parser.add_argument('--samples_per_iter', default=100, type=int,
                        help='Number of samples used during each iteration.')
    parser.add_argument('--start_iter', default=0, type=int, help='Iteration number to continue.')
    parser.add_argument('--max_iter', default=30, type=int, help='Number of training iterations.')
    parser.add_argument('--eval_only', default=False, action='store_true', help='Overwrite existing results.')
    parser.add_argument('--only_unique', default=False, action='store_true', help='Train only on unique samples.')
    parser.add_argument('--skip_first', default=False, action='store_true', help='Skip first training and propagation.')
    parser.add_argument('--cpu_only', default=False, action='store_true', help='Whether only the cpu or not.')
    parser.add_argument('--gpu_id', default=0, type=int, help='GPU id.')
    parser.add_argument('--overlap_thrs', default=0.1, type=float,
                        help='Threshold used to filter out intertwined instances.')
    return parser.parse_args()


# CUDA_VISIBLE_DEVICES=1 python iterative_training.py -e iterunique20210520 -m /media/hdd2/lkopi/datasets/rats/augmentation_test/detectron2_out/00012_instonly -gt /media/hdd2/lkopi/datasets/rats/tracking_test/test_videos/selected_data -gta /media/hdd2/lkopi/datasets/rats/tracking_test/test_videos/selected_data/annotations/selected_train_instonly.json -p /media/hdd2/lkopi/datasets/rats/tracking_test/iterunique20210520 --max_iter 0 --only_unique

def run(exp_name, model_dir, gt_dir, gt_annot, prop_dir, samples_per_iter=100, start_iter=0, max_iter=30,
        eval_only=False, only_unique=False, skip_first=False, cpu_only=False, gpu_id=0, overlap_thrs=0.1):
    model_parent_dir = os.path.dirname(model_dir)  # "/media/hdd2/lkopi/datasets/rats/augmentation_test/detectron2_out"
    model_name = os.path.basename(model_dir)  # "00012_instonly"

    n_samples = samples_per_iter
    exp_name = "{}_s{}".format(exp_name, samples_per_iter)
    seed = 42
    random.seed(seed)

    eval_dir = os.path.join(gt_dir, 'images')
    mask_dir = os.path.join(gt_dir, 'masks')
    annot_dir = os.path.dirname(gt_annot)
    annot_fn = os.path.basename(gt_annot)

    train_detectron = "CUDA_VISIBLE_DEVICES=" + str(gpu_id) + \
                      " python train_detectron2.py -tr {} -o {} --inst_only --eval_dir " + \
                      eval_dir + " --n_iter {} --resume"
    if cpu_only:
        train_detectron += " --cpu_only"

    if start_iter == 0:
        prev_out = model_dir
        prev_annot = None
    else:
        prev_exp_name = "{}_{}_iter{}".format(model_name, exp_name, start_iter - 1)
        prev_out = os.path.join(model_parent_dir, prev_exp_name)
        prev_annot = os.path.join(annot_dir, "{}_{}".format(prev_exp_name, annot_fn))
        n_samples = len(collect_prev_fns(prev_annot))

    for i in range(start_iter, max_iter + 1):
        curr_exp_name = "{}_{}_iter{}".format(model_name, exp_name, i)
        curr_out = os.path.join(model_parent_dir, curr_exp_name)

        if not os.path.exists(curr_out):
            os.system("cp -r {} {}".format(prev_out, curr_out))
            os.system("rm -rfd {}/eval_results".format(curr_out))
        print(curr_out)

        # Finetune detectron2
        print(n_samples)
        if 0 == i or eval_only or (skip_first and i == start_iter):
            os.system(train_detectron.format(prev_annot, curr_out, 50000 + i * 30 * n_samples).replace('--resume',
                                                                                                       '--eval_only'))
        else:
            if n_samples < 5:
                n_samples = 5
            elif n_samples < 30:
                n_samples = 30
            os.system(train_detectron.format(prev_annot, curr_out, 50000 + i * 30 * n_samples))

        # Propagate results
        out_dir = '{}/{}_overlap_thrs{}'.format(prop_dir, curr_exp_name, overlap_thrs)
        if not (skip_first and i == start_iter):
            propagate_detectron2(curr_out, out_dir, overlap_thrs=overlap_thrs)

        # Create a dataset for the next iteration: select bad samples
        curr_annot = os.path.join(annot_dir, "{}_{}".format(curr_exp_name, annot_fn))
        if not eval_only:
            n_samples = select_bad_samples(curr_annot, out_dir, eval_dir, gt_annot, samples_per_iter,
                                           seed=None, prev_fn=None if not only_unique or i == 0 else prev_annot)
            prev_annot = curr_annot

        # Evaluate it
        ts, iou = evaluate_results(out_dir, mask_dir)
        print(curr_exp_name, ts, iou)

        # Terminate training if the results are good
        if ts == 0:
            break

        # Initialization for the next iteration
        prev_out = curr_out


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
