#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import math
import multiprocessing
import os

import cv2
import numpy as np
import skimage.measure
from tqdm import tqdm

import utils.io
import utils.visualization

# good background sampling frame intervals in ..._retrograd_study_20190415_160817: (0,500), (10580,10760)
# Algorithm is based on https://forums.fast.ai/t/part-3-background-removal-with-robust-pca/4286


def _fast_mode_bin_3d(xs, n_bins):
    """
    Compute multiple 3D histograms at once (up to several magnitudes faster than np.histogramdd() called in loop).
    Parameters:
        xs: ndarray(n_parallel, n_xs, 3) of uint
        n_bins: int
    Returns:
        mode_bin_idxs: ndarray(n_parallel, 3) of int32
    """
    assert xs.shape[2:] == (3,)
    assert n_bins < 64
    bin_size = int(256 / n_bins)
    n_flatbins = n_bins * n_bins * n_bins
    n_parallel = xs.shape[0]
    xs_bin_idxs = (xs // bin_size).astype(np.int32)  # (n_parallel, n_xs, 3) of int32
    xs_bin_flat_idxs = np.ravel_multi_index((xs_bin_idxs[:, :, 0], xs_bin_idxs[:, :, 1], xs_bin_idxs[:, :, 2]),
                                            dims=(n_bins, n_bins, n_bins))  # (n_parallel, n_xs) of int32
    xs_par_idxs = np.arange(n_parallel, dtype=np.int32)  # (n_parallel,) of int32
    xs_par_idxs = np.broadcast_to(xs_par_idxs[:, None], shape=xs_bin_flat_idxs.shape)  # (n_parallel, n_xs) of int32
    xs_bincounts = np.zeros((n_parallel, n_flatbins), dtype=np.int32)  # (n_parallel, n_flatbins)
    np.add.at(xs_bincounts, (xs_par_idxs, xs_bin_flat_idxs), 1)
    # xs_bincounts[(xs_par_idxs, xs_bin_flat_idxs)] += 1, unbuffered
    xs_modebin_flat_idxs = np.argmax(xs_bincounts, axis=1)  # (n_parallel,) of int32, flatbin idxs
    xs_modebin_idxtup = np.unravel_index(xs_modebin_flat_idxs, shape=(n_bins, n_bins, n_bins))
    mode_bin_idxs = np.stack(xs_modebin_idxtup, axis=-1)
    return mode_bin_idxs


def separate_mode(ims, im_a=None):
    """
    Parameters:
        ims: ndarray(n_fr, sy, sx, n_ch) of uint8
        im_a: None OR ndarray(sy, sx, n_ch) of uint8; if given, mode is not calculated, im_a is used as the background
    Returns:
        im_a: ndarray(n_fr, sy, sx, n_ch) of uint8
        ims_e: ndarray(n_fr, sy, sx, n_ch) of uint8
    """
    n_bins = 32
    assert ims.dtype == np.uint8
    assert ims.shape[-1] == 3
    if im_a is None:
        orig_shape = ims.shape
        ims = ims.reshape((ims.shape[0], -1, ims.shape[-1]))  # (n_fr, sy*sx, n_ch) of uint8
        mode_bin_idxs = _fast_mode_bin_3d(ims.transpose((1, 0, 2)), n_bins)  # (sy*sx, 3) of int32
        bin_size = int(256 / n_bins)
        # bin_edges = np.append(np.arange(n_bins) * bin_size, 255)  # (n_bins+1)
        bin_centers = (np.arange(n_bins) * bin_size + (bin_size / 2)).astype(np.float32)  # (n_bins)

        # fg: abs(mode_bin_center - pix)
        im_a_fl = bin_centers[mode_bin_idxs]
        im_a_fl = im_a_fl.reshape(orig_shape[1:])
        im_a = im_a_fl.astype(np.uint8, copy=False)
        ims = ims.reshape(orig_shape)
    else:
        assert im_a.shape == ims.shape[1:]
        assert im_a.dtype == np.uint8
        im_a_fl = im_a.astype(np.float32, copy=False)

    ims_e = np.fabs(im_a_fl - ims.astype(np.float32)).astype(np.uint8)
    return im_a, ims_e


def get_rat_regions(im_e, prev_reg_data, brightness_thres, n_objs, min_ratio_to_biggest, apply_erode_intensity=0):
    """
    Searches for image regions in a brightness-thresholded image.
    Returns the centroid, bbox and mask of the top 'n_objs' biggest regions 
        which have a higher size ratio to the largest region than 'min_ratio_to_biggest'.
    Parameters:
        im_e: ndarray(sy, sx, n_ch) of uint8
        prev_reg_data: None OR same type as 'regions_data'
        brightness_thres: float; thresholding ims_e for pixels brighter than this limit
        n_objs: int
        min_ratio_to_biggest: float
        apply_erode_intensity: int
    Returns:
        regions_data: list(n_big_objs_found) of 
                            tuple(cy, cx, bbox_miny, minx, maxy, maxx, mask:ndarray(bbox_h, bbox_h));
                                     n_big_objs_found <= n_objs;

    """
    assert im_e.shape[2:] == (3,)
    assert 0. <= brightness_thres < 255.
    assert n_objs >= 1
    assert 0. <= min_ratio_to_biggest < 1.

    # threshold image
    im_e = np.amax(im_e, axis=-1)  # (sy, sx) of uint8
    im_e = (im_e > brightness_thres)  # (sy, sx) of bool_

    # erode mask optionally
    if apply_erode_intensity > 0:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (apply_erode_intensity, apply_erode_intensity))
        im_e = cv2.erode(im_e.astype(np.uint8), erode_kernel)

    # find connected foreground regions
    im_e, n_labels = skimage.measure.label(im_e, return_num=True)
    rprops = skimage.measure.regionprops(im_e, cache=True)  # list of regionprops

    # select top 'n_objs' biggest regions that are big enough
    biggest_reg_area = np.amax([rprop.area for rprop in rprops])
    min_obj_area = float(min_ratio_to_biggest) * biggest_reg_area
    biggest_reg_idxs = np.argsort([rprop.area for rprop in rprops])[-n_objs:][::-1]
    assert 1 <= biggest_reg_idxs.shape[0] <= n_objs

    # get cetroids for selected regions
    regions_data = []
    for reg_idx in biggest_reg_idxs:
        if rprops[reg_idx].area < min_obj_area:
            continue
        cy, cx = rprops[reg_idx].centroid
        miny, minx, maxy, maxx = rprops[reg_idx].bbox
        mask = rprops[reg_idx].image
        regions_data.append((cy, cx, miny, minx, maxy, maxx, mask))

    # tracking - only implemented for n_objs == 2 
    #   (switch order of objects in regions_data if mean obj distances show a swap in order compared to prev_reg_data)
    if (prev_reg_data is not None) and (len(prev_reg_data) == 2) and (len(regions_data) == 2):
        meandist_no_switch = (math.sqrt((prev_reg_data[0][0] - regions_data[0][0]) ** 2. +
                                        (prev_reg_data[0][1] - regions_data[0][1]) ** 2.) +
                              math.sqrt((prev_reg_data[1][0] - regions_data[1][0]) ** 2. +
                                        (prev_reg_data[1][1] - regions_data[1][1]) ** 2.)) * 0.5
        meandist_switch = (math.sqrt((prev_reg_data[0][0] - regions_data[1][0]) ** 2. +
                                     (prev_reg_data[0][1] - regions_data[1][1]) ** 2.) +
                           math.sqrt((prev_reg_data[1][0] - regions_data[0][0]) ** 2. +
                                     (prev_reg_data[1][1] - regions_data[0][1]) ** 2.)) * 0.5
        if meandist_switch < meandist_no_switch:
            regions_data = regions_data[::-1]

    return regions_data


def render_rat_regions(im_e, regions_data, colors_bgr8_tup, render_centroid=True, centroid_radius=5,
                       render_mask=True, mask_alpha=0.5, render_info=True):
    """
    Renders centroids and/or masks over image.
    Parameters:
        im_e: ndarray(sy, sx, n_ch) of uint8
        regions_data: <same as get_rat_regions() output format>
        colors_bgr8_tup: list of color tuples (bgr uint8) for each object
        render_centroid: bool
        centroid_radius: int
        render_mask: bool
        mask_alpha: float
        render_info: bool
    Returns:
        im_e: ndarray(sy, sx, n_ch) of uint8
    """
    assert len(colors_bgr8_tup) >= len(regions_data)
    colors_arr = np.array(colors_bgr8_tup, dtype=np.uint8)
    assert colors_arr.shape[1:] == (3,)
    if render_mask:
        mask = np.zeros_like(im_e)
        for reg_idx in range(len(regions_data)):
            cy, cx, miny, minx, maxy, maxx, mask_in_bbox = regions_data[reg_idx]
            mask[miny:maxy, minx:maxx, :] += mask_in_bbox[:, :, None] * colors_arr[reg_idx]
        im_e = cv2.addWeighted(im_e, 1. - mask_alpha, mask, mask_alpha, 0.)

    if render_centroid:
        for reg_idx in range(len(regions_data)):
            cy, cx, miny, minx, maxy, maxx, mask_in_bbox = regions_data[reg_idx]
            color_tup = (int(colors_arr[reg_idx][0]), int(colors_arr[reg_idx][1]), int(colors_arr[reg_idx][2]))
            cv2.circle(im_e, (int(cx), int(cy)), radius=centroid_radius, color=color_tup, thickness=-1)

    if render_info:
        if len(regions_data) == 2:
            text = 'SEPARATED'
            text_color = (0, 255, 0)
        elif len(regions_data) == 1:
            text = 'ANNOTATION NEEDED'
            text_color = (64, 64, 192)
        elif len(regions_data) == 0:
            text = 'ANNOTATION NEEDED (NONE FOUND)'
            text_color = (0, 0, 255)
        else:
            text = 'MORE THAN 2 OBJECTS'
            text_color = (255, 0, 255)
        cv2.putText(im_e, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, text_color, 2)

    return im_e


def read_frames_from_video_wrapper(params):
    return utils.io.read_frames_from_video(*params)[0]


def convert_bbox_format(bbox, input_format='xyxy', output_format='rcrc'):
    if input_format == output_format:
        return bbox
    if input_format in ['xyxy', 'rcrc']:
        if output_format in ['xyxy', 'rcrc']:
            return np.asarray(bbox)[[1, 0, 3, 2]].tolist()
    raise NotImplementedError()


def split_and_crop_video(video_in, out_dir, n_sequences=16, crop_bbox_xyxy=(0, 0, 640, 420),
                         start_frame=0, end_frame=None, save_video=False, n_processes=-1):  #TODO: visualize flag
    utils.io.make_directory(out_dir)
    img_out_format = os.path.join(out_dir, 'seq{:03d}')
    video_out_format = os.path.join(out_dir, 'seq{:03d}.avi')
    n_frames = utils.io.read_video_length(video_in)
    if end_frame is not None:
        n_frames = min(end_frame, n_frames)
    slice_length = 102
    seq_length = int(np.ceil(n_frames / n_sequences / slice_length)) * slice_length

    crop_bbox_rcrc = convert_bbox_format(crop_bbox_xyxy, input_format='xyxy', output_format='rcrc')

    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(min(n_processes, n_sequences)) as pool:
        for seq_id, frame_idx in enumerate(range(start_frame, n_frames, seq_length)):
            print("    at frame idx#" + str(frame_idx))
            with tqdm(total=(seq_length+slice_length-1) // slice_length) as pbar:
                for slice_idx, ims in enumerate(pool.imap(read_frames_from_video_wrapper,
                                                zip(itertools.repeat(video_in),
                                                    range(frame_idx, min(frame_idx+seq_length, n_frames), slice_length),
                                                    itertools.repeat(slice_length),
                                                    itertools.repeat((420, 640)),
                                                    itertools.repeat(crop_bbox_rcrc)))):

                    utils.io.save_images(img_out_format.format(seq_id), ims, start_idx=slice_idx*slice_length)
                    pbar.update()
            if save_video:
                ims = utils.io.read_arrays(utils.io.list_directory(img_out_format.format(seq_id)))
                utils.io.save_video(video_out_format.format(seq_id), ims, fps=30)


def run(video_in,
        out_dir,
        save_video=True,
        save_images=True,
        start_frame=0,
        end_frame=30000,
        crop_bbox_xyxy=(0, 0, 640, 420),
        compute_background_only_once=True,  # if True, bg of the first slice is used for all further slices
        compute_slice_len=300,
        brightness_thres=64):

    utils.io.make_directory(out_dir)
    video_out_format = os.path.join(out_dir, 'rat_sepbg{}.avi')
    img_out = os.path.join(out_dir, 'images')
    mask_out = os.path.join(out_dir, 'masks')
    bg_out = os.path.join(out_dir, 'bg')
    utils.io.make_directories(img_out, mask_out, bg_out)

    end_frame = min(end_frame, utils.io.read_video_length(video_in))
    assert start_frame < end_frame
    rat_colors_bgr = [(255, 0, 255), (255, 255, 0)]
    regions_data = None
    curr_bg = None
    frame_idx = start_frame
    crop_bbox_rcrc = convert_bbox_format(crop_bbox_xyxy, input_format='xyxy', output_format='rcrc')
    while frame_idx < end_frame:
        print("    at frame idx#" + str(frame_idx))
        ims = utils.io.read_frames_from_video(video_in, start_pos=frame_idx, n_frames=compute_slice_len,
                                              shape=(420, 640), crop_rcrc=crop_bbox_rcrc)[0]
        frame_idx += len(ims)

        # generate foreground (and background) images
        if (curr_bg is not None) and (compute_background_only_once or (ims.shape[0] < compute_slice_len)):
            _, ims_e = separate_mode(ims, im_a=curr_bg)
        else:
            curr_bg, ims_e = separate_mode(ims)

        # find rats in foreground image
        masked_ims = []
        simple_mask_ims = []
        rats_joint = []
        for im_e in ims_e:
            regions_data = get_rat_regions(im_e, regions_data, brightness_thres=brightness_thres,
                                           n_objs=2, min_ratio_to_biggest=0.3)  # (n_objs_found, 2:[y,x])

            rats_joint.append(len(regions_data) < 2)
            masked_im = np.zeros_like(im_e)
            masked_im = render_rat_regions(masked_im, regions_data, colors_bgr8_tup=rat_colors_bgr,
                                           render_centroid=False, centroid_radius=5,
                                           render_mask=True, mask_alpha=1., render_info=True)
            if save_images:
                simple_mask_im = np.zeros_like(im_e)
                simple_mask_im = render_rat_regions(simple_mask_im, regions_data,
                                                    colors_bgr8_tup=[(255, 255, 255), (127, 127, 127)],
                                                    render_centroid=False, centroid_radius=5,
                                                    render_mask=True, mask_alpha=1., render_info=False)
                simple_mask_im = np.amax(simple_mask_im, axis=-1)
                simple_mask_ims.append(simple_mask_im)

            masked_ims.append(masked_im)

        # write images to video
        frame_offset = frame_idx - ims.shape[0] - start_frame
        bgs = np.empty_like(ims)
        bgs[:] = curr_bg
        if save_video:
            tmp = utils.visualization.concat_images([ims, bgs, ims_e, np.asarray(masked_ims)], n_rows=2)
            utils.io.save_video(video_out_format.format(frame_offset), tmp, fps=30)

        if save_images:
            utils.io.save_images(img_out, ims, start_idx=frame_offset)
            utils.io.save_images(mask_out, np.asarray(simple_mask_ims), start_idx=frame_offset)
            utils.io.save_images(bg_out, bgs, start_idx=frame_offset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--video_in', required=True, help='Input video.')
    parser.add_argument('-o', '--out_dir', required=True, help='Directory to save the outputs.')
    parser.add_argument('-c', '--crop_bbox_rcrc', nargs='+', type=int, default=(160, 320, 580, 960),
                        help='Coordinates to crop the video, in the following order: \
                              top-left-row, top-left-col, bottom-right-row, bottom-right-col.')
    parser.add_argument('-s', '--start_frame', type=int, default=20, help='start_frame frame.')
    parser.add_argument('-n', '--end_frame', type=int, default=30000, help='Last frame.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
