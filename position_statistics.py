import itertools
from functools import partial
import multiprocessing
import os

import cv2
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import utils.io
import utils.segmentation
import utils.visualization
import utils.keypoints
import utils.utils


def run(mask_dir_base, switched_sequences=None, fn_suffix="",
        inner_box_xyxy=(213, 12, 408, 402), inner_box_size_in_meters=(0.4, 0.8), border_ratio=0.25,
        idle_threshold=0.05, min_duration=2, fps=30, n_processes=-1, chunksize=100, img_dir_base=None):

    seqs = utils.io.list_directory(mask_dir_base, only_dirs=True, full_path=False)
    mask_dirs = [os.path.join(mask_dir_base, seq) for seq in seqs]

    n_processes = utils.utils.determine_n_processes(n_processes, len(seqs))
    is_switched = utils.utils.create_switched_mask(switched_sequences, seqs).tolist()
    print(is_switched)

    with multiprocessing.Pool(processes=min(n_processes, len(mask_dirs))) as pool:
        with tqdm(total=len(mask_dirs)) as pbar:
            for _ in pool.starmap(partial(process_sequence,
                                          idle_threshold=idle_threshold,
                                          inner_box_size_in_meters=inner_box_size_in_meters,
                                          inner_box_xyxy=inner_box_xyxy, border_ratio=border_ratio,
                                          min_duration=min_duration, fps=fps, chunksize=chunksize),
                                  zip(mask_dirs, is_switched)):
                pbar.update()

    positions_csv, speed_csv = merge_sequences(mask_dir_base, mask_dirs, fn_suffix=fn_suffix)

    if img_dir_base is not None:
        plot_speed(speed_csv, idle_threshold=idle_threshold)

        img_dirs = utils.io.list_directory(img_dir_base, only_dirs=True)
        box_coords = calculate_middle_coordinates(inner_box_xyxy, border_ratio=border_ratio)
        visualize_position(positions_csv, img_dirs, box_coords, fps=fps, n_processes=n_processes)


def merge_sequences(out_dir, mask_dirs, fn_suffix=""):
    positions_fns = [f"{mask_dir}_positions.npy" for mask_dir in mask_dirs]
    speed_fns = [f"{mask_dir}_speed.npy" for mask_dir in mask_dirs]
    stat_fns = [f"{mask_dir}_stats.json" for mask_dir in mask_dirs]

    merge_arrays(f"{out_dir}{fn_suffix}_positions.xlsx", positions_fns)
    merge_arrays(f"{out_dir}{fn_suffix}_speed.xlsx", speed_fns)
    merge_statistics(f"{out_dir}{fn_suffix}_statistics.json", stat_fns)

    statistics_to_xlsx(f"{out_dir}{fn_suffix}_statistics.json")

    return f"{out_dir}{fn_suffix}_positions.xlsx", f"{out_dir}{fn_suffix}_speed.xlsx"


def merge_arrays(out_fn, fns):
    data = utils.io.read(fns[0])
    for fn in fns[1:]:
        data = np.concatenate((data, utils.io.read(fn)), axis=1)
    n_frames = data.shape[1]
    if data.ndim < 3:
        data = data[..., None]
    utils.io.save(out_fn, np.transpose(data, (1, 0, 2)).reshape(n_frames, -1))


def merge_statistics(out_fn, fns):
    merged_stats = utils.io.read(fns[0])
    for fn in fns[1:]:
        current_stats = utils.io.read(fn)
        for instance_id in merged_stats.keys():
            n_frames = merged_stats[instance_id]['stats']['n_frames']
            for category in merged_stats[instance_id].keys():
                for stat_type in merged_stats[instance_id][category].keys():
                    if stat_type == 'timesteps':
                        current_stats[instance_id][category][stat_type] = \
                            [ts + n_frames for ts in current_stats[instance_id][category][stat_type]]
                    merged_stats[instance_id][category][stat_type] += current_stats[instance_id][category][stat_type]
                    if not isinstance(merged_stats[instance_id][category][stat_type], list) \
                            and stat_type not in ['n_samples', 'n_frames']:
                        merged_stats[instance_id][category][stat_type] /= 2
    utils.io.save(out_fn, merged_stats)


def statistics_to_xlsx(json_fn):
    stats = utils.io.read(json_fn)

    max_len = 0
    data = {}
    for instance_id in stats.keys():
        for category in stats[instance_id].keys():
            for stat_type in stats[instance_id][category].keys():
                if stat_type not in {'timesteps', 'durations'}:
                    continue
                tmp = stats[instance_id][category][stat_type]
                if stat_type == 'timesteps':
                    tmp = [e/30 for e in tmp]
                max_len = max(max_len, len(tmp))
                data[f"rat{instance_id} {category} {stat_type}"] = tmp

    for k in data:
        length = len(data[k])
        if length < max_len:
            data[k] += ['']*(max_len-length)
            print(k, len(data[k]))

    df = pd.DataFrame(data)
    df.to_excel(json_fn.replace('.json', '.xlsx'), index=None, header=True)


def plot_speed(speed_csv, idle_threshold=0.05, apply_smoothing=True, window=3):
    speed = utils.io.read(speed_csv, use_numpy=True)
    assert speed.shape[-1] == 2, speed.shape
    speed = speed[:len(speed)//30*30].reshape(-1, 30, 2).mean(axis=1)
    if apply_smoothing:
        speed[:, 0] = pd.Series(speed[:, 0]).rolling(window=window, min_periods=1,
                                                     center=True, closed='both').mean().values
        speed[:, 1] = pd.Series(speed[:, 1]).rolling(window=window, min_periods=1,
                                                     center=True, closed='both').mean().values

    plt.figure(figsize=(12*(len(speed)//120), 6))
    plt.plot(speed[:, 0])
    plt.plot(speed[:, 1])
    plt.plot([idle_threshold] * len(speed))
    plt.xticks(np.arange(0, len(speed), 10))
    plt.grid(True, axis='y')
    plt.margins(x=0)
    plt.savefig(speed_csv.replace('.xlsx', '.png'), bbox_inches='tight')


def visualize_position(position_csv, img_dirs, box_coords, fps=30, n_processes=-1):
    img_fns = []
    if isinstance(img_dirs, str):
        img_dirs = [img_dirs]
    for img_dir in img_dirs:
        img_fns += utils.io.list_directory(img_dir, extension='.png', sort_key=utils.io.filename_to_number)
    positions = utils.io.read(position_csv, use_numpy=True)
    memory = 30
    sliding_positions = np.lib.stride_tricks.sliding_window_view(positions, (memory, 4)).reshape((-1, memory, 4))

    initial_frames = ((img_fns[current_idx], sliding_positions[0][:current_idx+1], box_coords)
                      for current_idx in range(memory))
    subsequent_frames = ((img_fns[current_idx], sliding_positions[current_idx - memory], box_coords)
                         for current_idx in range(memory, min(len(img_fns)-1, len(sliding_positions))))
    frames = itertools.chain(initial_frames, subsequent_frames)

    img_shape = utils.io.read(img_fns[0]).shape

    utils.io.save_video_parallel(position_csv.replace('.xlsx', '.mp4'), visualize_frame_wrapper, frames, img_shape,
                                 fps=fps, length=len(positions), chunksize=1, n_processes=n_processes)


def visualize_frame_wrapper(params):  # TODO: decorator wrapper
    return visualize_frame(*params)


def visualize_frame(img_fn, positions, box_coords):
    # print(img_fn)  # , positions, box_coords)
    w1, h1, w2, h2 = box_coords
    img = utils.io.read(img_fn)
    for position in positions:
        img = cv2.circle(img, (int(position[1]), int(position[0])), radius=5, color=(0, 0, 255), thickness=-1)
        img = cv2.circle(img, (int(position[3]), int(position[2])), radius=5, color=(0, 255, 0), thickness=-1)
    img = cv2.rectangle(img, (w1, h1), (w2, h2), color=(255, 0, 0), thickness=3)
    return img


def process_sequence(mask_dir, is_switched, idle_threshold, inner_box_size_in_meters, inner_box_xyxy,
                     border_ratio, min_duration, fps, chunksize):
    positions = estimate_position(mask_dir, is_switched=is_switched, chunksize=chunksize)
    px2m = check_inner_box_size(inner_box_xyxy, inner_box_size_in_meters)
    speed = estimate_speed(positions, px2m=px2m)
    statistics = calculate_all_statistics(positions, speed,
                                          idle_threshold=idle_threshold, min_duration=min_duration, fps=fps,
                                          inner_box_xyxy=inner_box_xyxy, border_ratio=border_ratio)
    utils.io.save(f"{mask_dir}_positions.npy", positions)
    utils.io.save(f"{mask_dir}_speed.npy", speed)
    utils.io.save(f"{mask_dir}_stats.json", statistics)


def estimate_position(mask_dir, is_switched=False, chunksize=100, window=5):
    fns = utils.io.list_directory(mask_dir, sort_key=utils.io.filename_to_number, extension='.png')

    positions = np.zeros((2, len(fns), 2), dtype=float)
    current_idx, finished, channel_ids = 0, False, None
    while current_idx < len(fns):
        masks = utils.io.read_arrays(fns[current_idx:current_idx+chunksize])
        if channel_ids is None:
            channel_ids = np.unique(utils.segmentation.multi2singlechannel(masks[:1]))[1:] - 1
            if len(channel_ids) < 2:
                print(f"[WARNING] {mask_dir}: {channel_ids}")
                return positions
            # assert 2 == len(channel_ids), f"{mask_dir}: {channel_ids}"
        masks = 255*(masks > 0)

        for frame_id in range(len(masks)):
            coords = utils.keypoints.heatmap2keypoints(masks[frame_id], round_coords=False)
            for inst_id, channel_id in enumerate(channel_ids):
                positions[inst_id, current_idx+frame_id] = coords[channel_id]

        current_idx += chunksize

    if is_switched:
        positions = positions[::-1, ...]

    # source: https://stackoverflow.com/a/30141358
    # parameters: https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html
    if 1 < window:
        for inst_id in range(2):
            positions[inst_id, :, 0] = pd.Series(positions[inst_id, :, 0]).rolling(window=window, min_periods=1,
                                                                                   center=True, closed='both').mean().values
            positions[inst_id, :, 1] = pd.Series(positions[inst_id, :, 1]).rolling(window=window, min_periods=1,
                                                                                   center=True, closed='both').mean().values

    return positions


def check_inner_box_size(inner_box_xyxy, inner_box_size_in_meters, max_error_ratio=0.2):
    w1, h1, w2, h2 = inner_box_xyxy
    width_in_meters, height_in_meters = inner_box_size_in_meters

    px2m_width = width_in_meters / (w2-w1)
    px2m_height = height_in_meters / (h2-h1)

    error_ratio = np.abs(px2m_width / px2m_height - 1)
    assert error_ratio < max_error_ratio, \
        f"The position {inner_box_xyxy} and the size {inner_box_size_in_meters} of the inner box does not match!"

    return (px2m_width + px2m_height) / 2


def estimate_speed(positions, window=5, px2m=0.8/390, fps=30):
    speed = np.sqrt(np.sum((positions[:, 1:] - positions[:, :-1]) ** 2, axis=-1)).astype(float)
    for inst_id in range(2):
        speed[inst_id, :] = pd.Series(speed[inst_id, :]).rolling(window=window, min_periods=1,
                                                                 center=True, closed='both').mean().values
    speed *= px2m * fps
    return speed


def calculate_statistics(data, min_duration=2, fps=30):
    durations = data[1:] - data[:-1] - 1
    timesteps = data[:-1][durations > 2]
    durations = durations[durations > min_duration] / fps

    out = {
        'timesteps': timesteps.tolist(),
        'durations': durations.tolist(),
        'n_samples': len(durations),
        'sum': np.sum(durations) if 0 < len(durations) else 0,
        'mean': np.mean(durations) if 0 < len(durations) else 0,
        'median': np.median(durations) if 0 < len(durations) else 0,
        'std': np.std(durations) if 0 < len(durations) else 0,
    }
    return out


def calculate_idle_statistics(speed, idle_threshold=0.05, min_duration=2, fps=30, **kwargs):
    in_motion = np.where(speed >= idle_threshold)[0]
    return calculate_statistics(in_motion, min_duration=min_duration, fps=fps)


def calculate_middle_coordinates(inner_box_xyxy, border_ratio=0.25):
    w1, h1, w2, h2 = inner_box_xyxy

    width = w2 - w1
    height = h2 - h1
    border_width = int(width * border_ratio)
    border_height = int(height * border_ratio)

    h1 += border_height
    h2 -= border_height
    w1 += border_width
    w2 -= border_width

    return w1, h1, w2, h2


def calculate_in_the_middle_statistics(position, inner_box_xyxy, border_ratio=0.25, min_duration=2, fps=30, **kwargs):
    w1, h1, w2, h2 = calculate_middle_coordinates(inner_box_xyxy, border_ratio)
    in_the_middle = np.logical_and(np.logical_and(h1 < position[:, 0], position[:, 0] < h2),
                                   np.logical_and(w1 < position[:, 1], position[:, 1] < w2))
    in_the_middle = np.where(~in_the_middle)[0]
    return calculate_statistics(in_the_middle, min_duration=min_duration, fps=fps)


def calculate_all_statistics(positions, speed, **kwargs):
    statistics = {}
    for inst_id in range(len(positions)):
        statistics[str(inst_id)] = {
            'stats': {
                'n_frames': positions.shape[1],
            },
            'idle': calculate_idle_statistics(speed[inst_id], **kwargs),
            'speed': {
                'mean': np.mean(speed[inst_id]),
                'median': np.median(speed[inst_id]),
                'std': np.std(speed[inst_id]),
            },
            'in_the_middle': calculate_in_the_middle_statistics(positions[inst_id], **kwargs),
        }

    return statistics
