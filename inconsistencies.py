import multiprocessing
import os

import numpy as np
from tqdm import tqdm

import utils.io
import utils.segmentation
import utils.keypoints
import eval_propagation


def detect_inconsistencies(main_pred_dir, secondary_pred_dir):
    fns = utils.io.list_directory(main_pred_dir, sort_key=utils.io.filename_to_number, extension='.png')
    masks1 = utils.io.read_arrays(fns)
    masks1 = utils.segmentation.multi2singlechannel(masks1)
    inst1_ids = np.unique(masks1[:1])[1:]

    fns = utils.io.list_directory(secondary_pred_dir, sort_key=utils.io.filename_to_number, extension='.png')
    masks2 = utils.io.read_arrays(fns)
    # print(masks1.shape, masks2.shape)
    inst2_ids = np.unique(masks2[:1])[1:]

    print(main_pred_dir, inst1_ids, inst2_ids, masks2.shape)
    n_switches, _, switched_frames = eval_propagation.count_track_switches(masks2, masks1, return_switched_frames=True)
    return n_switches, switched_frames


def run(main_pred_dir, secondary_pred_dir):
    seqs = utils.io.list_directory(main_pred_dir, only_dirs=True, full_path=False)
    main_pred_dirs = [os.path.join(main_pred_dir, seq) for seq in seqs]
    secondary_pred_dirs = [os.path.join(secondary_pred_dir, seq) for seq in seqs]

    out = {}
    all_switches = 0
    all_switched = []
    with multiprocessing.Pool(processes=4) as pool:
        with tqdm(total=len(seqs)) as pbar:
            for idx, (n_switches, switched_frames) in enumerate(pool.starmap(detect_inconsistencies,
                                                                             zip(main_pred_dirs, secondary_pred_dirs))):
                all_switches += n_switches
                all_switched += switched_frames  # TODO: shift by idx*n_frames
                out[seqs[idx]] = {
                    'n_switches': n_switches,
                    'switched_frames': switched_frames,
                }
                pbar.update()
    out['n_switches'] = all_switches

    utils.io.save(f"{main_pred_dir}_inconsistencies.json", out)


if __name__ == '__main__':
    # main_pred_dir = '/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_1_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1'
    main_pred_dirs = [
        '/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_3_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1',
        #'/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA animals/1-Autism study_autistic animals_1_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1',
        #'/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA animals/1-Autism study_autistic animals_3_selected_instonly_eval_s100_i1000_iter0_model_only_clean/prop_results_overlap_thrs_0.1',
    ]
    for main_pred_dir in main_pred_dirs:
        secondary_pred_dir = main_pred_dir + '_old'
        run(main_pred_dir, secondary_pred_dir)
