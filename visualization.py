import multiprocessing
import os

import numpy as np
from tqdm import tqdm

import utils.io
import utils.segmentation
import utils.visualization
import utils.utils


def visualize_frame(img_fn, mask_fn, is_switched=False):
    imgs = utils.io.read(img_fn)[None]
    masks = utils.io.read(mask_fn)[None]

    channels = np.unique(utils.segmentation.multi2singlechannel(masks[:1]))[1:] - 1
    if len(channels) < 2:
        print(f"[WARNING] {mask_fn}: {channels} ({is_switched})")
        return imgs[0]
    if is_switched:
        channels = channels[::-1]

    imgs = utils.visualization.overlay_mask(imgs, masks[..., channels[0]], color=(0, 0, 255))
    imgs = utils.visualization.overlay_mask(imgs, masks[..., channels[1]], color=(0, 255, 0))

    return imgs[0]


def visualize_sequence(img_dir, mask_dir, out_fn, is_switched):
    img_fns = utils.io.list_directory(img_dir, extension='.png', sort_key=utils.io.filename_to_number)
    pred_fns = utils.io.list_directory(mask_dir, extension='.png', sort_key=utils.io.filename_to_number)

    # Visualize frames in batches using a generator to do it lazily
    frames = (visualize_frame(img_fns[current_idx], pred_fns[current_idx], is_switched=is_switched)
              for current_idx in range(min(len(img_fns), len(pred_fns))))
    img_shape = utils.io.read(img_fns[0]).shape
    utils.io.save_video(out_fn, frames, fps=30, shape=img_shape)


def visualize_predictions_memory_conservative(img_dir, mask_dir, switched_sequences=None, concat_videos=False,
                                              n_processes=-1, out_fn=None, seqs=None):
    if seqs is None:
        seqs = utils.io.list_directory(mask_dir, only_dirs=True, full_path=False)
    img_dirs = [os.path.join(img_dir, seq) for seq in seqs]
    mask_dirs = [os.path.join(mask_dir, seq) for seq in seqs]
    out_fns = [f"{md}.mp4" for md in mask_dirs]

    n_processes = utils.utils.determine_n_processes(n_processes, len(seqs))
    is_switched = utils.utils.create_switched_mask(switched_sequences, seqs).tolist()

    with multiprocessing.Pool(processes=min(n_processes, len(mask_dirs))) as pool:
        with tqdm(total=len(mask_dirs)) as pbar:
            for _ in pool.starmap(visualize_sequence, zip(img_dirs, mask_dirs, out_fns, is_switched)):
                pbar.update()

    if concat_videos:
        if out_fn is None:
            out_fn = f"{mask_dir}.mp4"
        utils.io.merge_videos(out_fn, out_fns)
