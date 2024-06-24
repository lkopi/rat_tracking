import os
import functools
import multiprocessing

from tqdm import tqdm

import utils.io


def run_gma(img_dir, gma_dir):
    curr_dir = os.getcwd()
    os.chdir(gma_dir)
    os.system("python gma_evaluate_directory.py --model checkpoints/gma-sintel.pth --path \"{}\"".format(img_dir))
    os.chdir(curr_dir)


def run(img_dir, gma_dir, n_processes=6):
    script_fn = os.path.join(gma_dir, 'gma_evaluate_directory.py')
    if not os.path.exists(script_fn):
        utils.io.copy_file('gma_evaluate_directory.py', script_fn)

    seq_dirs = utils.io.list_directory(img_dir, only_dirs=True)
    with multiprocessing.Pool(min(n_processes, len(seq_dirs))) as pool:
        with tqdm(total=len(seq_dirs)) as pbar:
            for _ in pool.imap_unordered(functools.partial(run_gma, gma_dir=gma_dir), seq_dirs):
                pbar.update()
