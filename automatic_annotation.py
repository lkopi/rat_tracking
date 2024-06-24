#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import math
import multiprocessing
import os
import shutil
from dataclasses import dataclass

import cv2
import numpy as np
import scipy.ndimage as ndimage
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import gaussian
from skimage.graph import route_through_array
from skimage.measure import (label,
                             regionprops)
from skimage.morphology import (closing, medial_axis,
                                remove_small_holes,
                                remove_small_objects)
from skimage.segmentation import active_contour, watershed

import utils.io
import utils.segmentation
from utils.logger import PlotLogger


def remove_holes_and_small_regions(mask):
    """Remove small holes and small regions.

    Parameters
    ----------
    mask : np.ndarray
        2-dimensional binary mask.

    Returns
    -------
    np.ndarray
        A 2-dimensional binary mask where small regions and holes are removed.
    """
    data = remove_small_holes(mask, connectivity=4)
    data = closing(data)
    # data = opening(data, square(3))
    data = remove_small_objects(data, 100)
    return data


def calculate_midline(distance_map, boundary_mask):
    """Given a distance map and a mask it calculates the midline.

    Parameters
    ----------
    distance_map : np.ndarray
        2-dimensional uint8 matrix.
    boundary_mask : np.ndarray
        2-dimensional binary mask.

    Returns
    -------
    np.ndarray
        A 2D array representing a line.
    """
    midline = (cv2.Laplacian(distance_map, cv2.CV_64F) < -0.8) * ~boundary_mask
    midline = remove_small_objects(midline, 10, connectivity=2)
    midline = medial_axis(midline)
    return midline  # (midline*255).astype(np.uint8)


def find_line_endings(mask):
    """Find the line endings of a line represented by a 2D array.

    Parameters
    ----------
    mask : np.ndarray
        2-dimensional binary image of a line.

    Returns
    -------
    np.ndarray
        N x 2 int array, where each row represents a coordinate of a line ending.
    """
    def line_end_filter(kernel):
        """Central pixel and just one other must be set to be a line end"""
        return kernel[4] == 1 and np.sum(kernel) == 2

    result = ndimage.generic_filter(mask, line_end_filter, (3, 3))
    return np.argwhere(result)


def create_markers(coordinates, shape):
    markers = utils.segmentation.coordinates_to_mask(coordinates, shape)
    markers = ndimage.label(markers)[0]
    label_ids = markers[np.round(coordinates[:, 0]).astype('int'), np.round(coordinates[:, 1]).astype('int')]
    return markers, label_ids


def calculate_path_cost(weights, start_coord, end_coord, return_path=False):
    path, cost = route_through_array(weights, start_coord, end_coord)
    if return_path:
        return cost, path
    return cost


def select_furthest_coordinates(coordinates, path_cost=None, max_step_cost=3, weights=None):
    def calculate_distance(coord_pair):
        c1, c2 = coord_pair[:2], coord_pair[2:]
        return calculate_path_cost(weights, c1, c2)

    assert (path_cost is not None and weights is None) or (path_cost is None and weights is not None)

    n_coords = len(coordinates)
    x1 = np.broadcast_to(coordinates[:, None, :], (n_coords, n_coords, 2))
    x2 = np.broadcast_to(coordinates, (n_coords, n_coords, 2))
    triu_x, triu_y = np.triu_indices(n_coords, k=1)
    pairs = np.dstack((x1, x2))[triu_x, triu_y, :].tolist()
    if weights is None:
        weights = max_step_cost - (max_step_cost-1) * path_cost
    dists = list(map(calculate_distance, pairs))
    max_pair = pairs[np.argmax(dists)]

    return np.array(max_pair).reshape(2, 2)


def find_closest_coordinate(coordinates, closest_to, furthest_from=None):
    assert coordinates.ndim == 2
    assert coordinates.shape[1] == 2
    assert len(closest_to) == 2
    assert furthest_from is None or len(furthest_from) == 2

    distance = np.linalg.norm(coordinates - np.array(closest_to).reshape(1, 2), ord=2, axis=1)
    if furthest_from is not None:
        distance -= np.linalg.norm(coordinates - np.array(furthest_from).reshape(1, 2), ord=2, axis=1)
    return coordinates[np.argmin(distance), :]


def draw_line(arr, start_coord, end_coord, value=255):
    start_coord = np.array(start_coord)
    end_coord = np.array(end_coord)
    diag_dist = int(round(np.abs(start_coord - end_coord).max() + 2))
    x_coords = np.linspace(start_coord[0], end_coord[0], diag_dist, endpoint=True, dtype=np.int32)
    y_coords = np.linspace(start_coord[1], end_coord[1], diag_dist, endpoint=True, dtype=np.int32)
    arr[y_coords, x_coords] = value


def get_line_coords(start_coord, end_coord):
    start_coord = np.array(start_coord)
    end_coord = np.array(end_coord)
    diag_dist = np.abs(start_coord - end_coord).max() + 1
    x_coords = np.linspace(start_coord[0], end_coord[0], diag_dist, endpoint=True, dtype=np.int32)
    y_coords = np.linspace(start_coord[1], end_coord[1], diag_dist, endpoint=True, dtype=np.int32)
    return x_coords, y_coords


def map_point_to_line(ap, a1, b1, a2, b2):
    """Calculate the value (bp) at the given point (ap) on the line defined by (a1,b1) and (a2,b2).

    Parameters
    ----------
    ap : float
        Position of the desired point.
    a1 : float
        Position of the first point.
    b1 : float
        Value of the first point
    a2 : float
        Position of the second point.
    b2 : float
        Value of the second point

    Returns
    -------
    float
        Value of the desired point.
    """
    if a2 < a1:
        a1, a2 = a2, a1
        b1, b2 = b2, b1
    assert a1 <= ap and ap <= a2

    r = ap / (a2-a1)
    bp = r*b1 + (1-r)*b2
    return bp


def process_masks(masks, vis_fn=None, params=None):
    """Annotate the given mask and save visualizations.

    Parameters
    ----------
    masks : list of np.ndarray
        A list of 2D binary masks.
    vis_fn : str, optional
        Filename to save the visualization, by default None
    params : AnnotParams
    """

    def print_log(msg):
        if params.verbose:
            tmp = vis_fn + ' ' if vis_fn is not None else ''
            print('{}{} {}'.format(tmp, i, msg))

    # Initialize
    if params is None:
        params = AnnotParams()

    keypoints = {}
    parts = np.zeros(masks[0].shape, dtype=np.uint8)
    instances = np.zeros(masks[0].shape, dtype=np.uint8)
    if len(masks) == 1: # TODO remove?
        return keypoints, parts, instances
    logger = PlotLogger(masks[0].shape) if vis_fn is not None else None

    # Process masks
    for i, mask in enumerate(masks):
        try:
            mask_clean = clean_mask(mask, params, logger=logger)
            instances += ((i+1) * mask_clean).astype(np.uint8)

            # Find keypoints
            head_coord_smooth, tail_base_coord, tail_end_coord, tail_coord_smooth, distance_map, tail_mask = find_keypoints(mask_clean, mask, params, logger=logger)
            keypoints[i] = {'head': list(map(int, head_coord_smooth)),
                            'tail_base': list(map(int, tail_base_coord)),
                            'tail': list(map(int, tail_end_coord))}

            # Segment rat parts
            parts += segment_bodyparts(distance_map, head_coord_smooth, tail_coord_smooth, mask_clean, tail_mask, params, logger=logger)

        except Exception as e:
            log_prefix = '[Skip mask]:'
            if not isinstance(e, AssertionError):
                log_prefix = '[ERROR]:'
                import traceback
                traceback.print_exc()
            print_log('{} {}'.format(log_prefix, e))
            if logger is not None:
                logger.plot(message=e)
        finally:
            if vis_fn is not None:
                base_fn, ext = os.path.splitext(vis_fn)
                fig_fn = base_fn + '_' + str(i) + ext
                logger.save_figure(fig_fn)

    print(keypoints)
    return keypoints, parts, instances


def segment_bodyparts(distance_map, head_coord_smooth, tail_coord_smooth, mask_clean, tail_mask, params, logger=None):
    # Segment rat parts
    max_dist_coord = np.unravel_index(np.argmax(distance_map, axis=None), distance_map.shape)
    dense_coords = np.vstack((head_coord_smooth, tail_coord_smooth, max_dist_coord))
    dense_markers, dense_label_ids = create_markers(dense_coords, mask_clean.shape)
    dense_head_id, dense_tail_id, dense_body_id = dense_label_ids

    densepose = watershed(-distance_map, dense_markers, mask=mask_clean, compactness=params.snd_ws_compactness)
    tail_densepose = (densepose == dense_tail_id)
    if params.keep_orig_tail:
        tail_densepose |= tail_mask
    body_densepose = (densepose == dense_body_id) & ~tail_densepose
    head_densepose = (densepose == dense_head_id)

    # Relabel
    densepose = np.zeros_like(densepose)
    densepose[tail_densepose] = 1
    densepose[body_densepose] = 2
    densepose[head_densepose] = 3

    if params.split_head_region:
        densepose = split_head_segment(densepose, head_densepose, head_coord_smooth, logger=logger)
    return densepose.astype(np.uint8)


def split_head_segment(densepose, head_densepose, head_coord_smooth, logger=None):
    # Split head region into two
    # Help: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html
    props = list(filter(lambda p: p.label == 3, regionprops(densepose)))
    props.sort(key=lambda p: p.area, reverse=True)
    props = props[0]

    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 - math.sin(orientation) * props.major_axis_length
    y1 = y0 - math.cos(orientation) * props.major_axis_length
    x2 = x0 + math.sin(orientation) * props.major_axis_length
    y2 = y0 + math.cos(orientation) * props.major_axis_length

    minr, minc, maxr, maxc = props.bbox
    if maxc - x1 > maxc - x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    deg = np.rad2deg(orientation)
    draw_line(head_densepose, (x1, y1), (x2, y2), False)

    head_parts = label(head_densepose.astype(np.uint8), connectivity=1)
    head_regions = regionprops(head_parts)
    head_regions.sort(key=lambda p: (p.centroid[0]-head_coord_smooth[0])**2 + (p.centroid[1]-head_coord_smooth[1])**2, reverse=True)
    if logger is not None:
        logger.plot(image=head_parts)
    assert 2 == len(head_regions), 'Unable to separate head into 2 regions, instead: {}'.format(len(head_regions))

    body_mask = np.zeros_like(head_densepose, dtype=np.bool)
    for p in head_regions[:-1]:
        h1, w1, h2, w2 = p.bbox
        tmp = body_mask.copy()
        tmp[h1:h2, w1:w2] = p.image
        body_mask |= tmp

    densepose = densepose.copy()
    densepose[body_mask] = 2
    if logger is not None:
        logger.plot(image=densepose)

    return densepose


def clean_mask(mask, params, logger=None):
    mask_clean = remove_holes_and_small_regions(mask) if params.clean else mask

    # Visualize partial results
    if logger is not None:
        boundary_clean = utils.segmentation.find_boundary_coordinates(mask_clean)
        boundary = utils.segmentation.find_boundary_coordinates(mask)
        logger.plot(coordinate_list=boundary_clean, colors=['red'], size=2)
        logger.plot(coordinate_list=boundary, colors=['blue'], size=1)

    # Check whether the rat has tail (in other words: not too convex)
    is_conv, ratio = utils.segmentation.is_convex(mask_clean, threshold=params.convexity_threshold, return_overlap=True)
    assert not is_conv, 'Too convex... {:.2f}'.format(ratio)
    return mask_clean


def find_keypoints(mask_clean, mask, params, logger=None):
    distance_map, skeleton, endings_skeleton, midline_smooth, endings_smooth, midline_merge, endings = estimate_midlines_and_endpoints(mask_clean, params, logger)

    head_coord_smooth, tail_coord_smooth, tail_base_coord, tail_mask = estimate_keypoint_locations(mask_clean, distance_map, skeleton, midline_smooth, endings_smooth, params)

    head_coord_smooth, tail_end_coord = improve_keypoints(head_coord_smooth, tail_coord_smooth, mask, endings, endings_smooth, endings_skeleton, midline_merge, params)

    # Visualize partial results
    if logger is not None:
        logger.plot(coordinate_list=[head_coord_smooth, tail_coord_smooth, tail_base_coord, tail_end_coord], colors=['green', 'red', 'magenta', 'yellow'], size=6, plot_as_dots=True)

    # Calculate body and tail length
    check_body_tail_ratio(midline_merge, tail_end_coord, tail_base_coord, head_coord_smooth, params)

    return head_coord_smooth, tail_base_coord, tail_end_coord, tail_coord_smooth, distance_map, tail_mask


def improve_keypoints(head_coord_smooth, tail_coord_smooth, mask, endings, endings_smooth, endings_skeleton, midline_merge, params):
    # Improve keypoints
    corners = corner_peaks(corner_harris(mask), min_distance=5)
    coords = select_furthest_coordinates(np.concatenate((endings, endings_smooth), axis=0), midline_merge, params.max_step_cost)
    orig_head_coord = list(find_closest_coordinate(coords, head_coord_smooth))
    coords = select_furthest_coordinates(np.concatenate((endings_skeleton, endings_smooth), axis=0), midline_merge, params.max_step_cost)
    tail_end_coord = list(find_closest_coordinate(coords, tail_coord_smooth))
    tail_end_coord = list(find_closest_coordinate(corners, tail_end_coord))
    if params.use_head_corner:
        head_coord_smooth = list(find_closest_coordinate(corners, head_coord_smooth))
    return head_coord_smooth, tail_end_coord


def estimate_midlines_and_endpoints(mask_clean, params, logger):
    # Estimate midline and calculate the distance map and the skeleton
    midline_smooth, distance_smooth, skeleton_smooth = estimate_midline(smooth_mask(mask_clean, params))
    midline, distance_map, skeleton = estimate_midline(mask_clean)

    # Find line ends of the midline
    endings_skeleton = find_line_endings(skeleton)
    endings = find_line_endings(midline)
    endings_smooth = find_line_endings(midline_smooth)
    # If midline cannot be found use the smooth midline instead
    if len(endings_smooth) < 2:
        midline_smooth = skeleton_smooth
        endings_smooth = find_line_endings(midline_smooth)
        if len(endings_smooth) < 2:
            midline_smooth, endings_smooth, skeleton_smooth, distance_smooth = midline, endings, skeleton, distance_map
    midline_merge = (0 < (midline + midline_smooth))

    # Visualize partial results
    if logger is not None:
        midline_coords, midline_coords_smooth = np.argwhere(midline_merge > 0), np.argwhere(midline_smooth > 0)
        logger.plot(coordinate_list=[midline_coords, midline_coords_smooth], colors=['grey', 'cyan'], size=1, plot_as_dots=True)
        logger.plot(coordinate_list=[endings, endings_smooth], colors=['grey', 'cyan'], size=4, plot_as_dots=True)

    # Check that endpoints are within the mask
    assert 2 <= len(endings_smooth), 'Not enough endpoints... orig:{}, snake:{}'.format(len(endings), len(endings_smooth))
    return distance_map, skeleton, endings_skeleton, midline_smooth, endings_smooth, midline_merge, endings


def estimate_keypoint_locations(mask_clean, distance_map, skeleton, midline_smooth, endings_smooth, params):
    # Get neck and tail base coordinates
    # dmap, skel = distance_map, skeleton
    # if params.use_snake_skel:
    #    dmap, skel = distance_smooth, skeleton_smooth
    potential_tail_base_coords = find_potential_tail_base_coordinates(distance_map, skeleton, use_median=params.use_median)

    # Prepare for watershed: put markers (head and tail points) on the image
    head_and_tail_coords = select_furthest_coordinates(endings_smooth, midline_smooth, params.max_step_cost)
    markers, _ = create_markers(head_and_tail_coords, mask_clean.shape)
    assert 2 <= np.count_nonzero(markers * mask_clean), 'Marker out of mask...'

    # Run watershed => smaller region (because the weight are small in that area) == tail, larger region == head
    labels = watershed(-distance_map, markers, mask=mask_clean, compactness=0.05)
    ids, counts = np.unique(labels, return_counts=True)
    ids, counts = ids[1:], counts[1:]
    head_id = max(zip(ids, counts), key=lambda x: x[1])[0]
    tail_id = min(zip(ids, counts), key=lambda x: x[1])[0]
    tail_mask = (labels == tail_id)

    # Get head, tail and tail base coordinates
    head_coord_smooth = list(np.argwhere(markers == head_id)[0])
    tail_coord_smooth = list(np.argwhere(markers == tail_id)[0])
    tail_base_coord = list(find_closest_coordinate(potential_tail_base_coords, tail_coord_smooth, head_coord_smooth))
    return head_coord_smooth, tail_coord_smooth, tail_base_coord, tail_mask


def check_body_tail_ratio(midline_merge, tail_end_coord, tail_base_coord, head_coord_smooth, params):
    dist_map = 3 - 2*(midline_merge)  # 2 - midline
    tail_length = calculate_path_cost(dist_map, tail_end_coord, tail_base_coord)
    body_length = calculate_path_cost(dist_map, tail_base_coord, head_coord_smooth)

    body_tail_ratio = min(tail_length, body_length) / max(tail_length, body_length)
    assert body_tail_ratio >= params.body_tail_ratio, 'Tail is too short... tail:{:.2f}, body:{:.2f}, ratio:{:.2f}'.format(tail_length, body_length, body_tail_ratio)


def find_potential_tail_base_coordinates(distance_map, skeleton, use_median=True):
    dist_on_skel = distance_map * skeleton
    dist_mean = np.mean(dist_on_skel[dist_on_skel > 0])
    dist_median = np.median(dist_on_skel[dist_on_skel > 0])
    dist_on_skel[dist_on_skel < dist_mean] = 0

    dist_measure = dist_median if use_median else dist_mean
    skeleton[dist_on_skel < dist_median] = 0
    skeleton[dist_on_skel < dist_measure] = 0
    potential_tail_base_coords = find_line_endings(skeleton)
    return potential_tail_base_coords


def smooth_mask(mask, params):
    if params.use_snakes:
        snake = apply_active_contour(mask, params.snake_alpha, params.snake_beta)
    else:
        snake = utils.segmentation.smooth_mask(mask, degree=params.smooth_degree, n_iter=params.smooth_iter)[0]  # Keep only the largest one
    return utils.segmentation.decode_mask(snake, mask.shape)


def apply_active_contour(mask, snake_alpha=0.01, snake_beta=0.3):
    boundary = utils.segmentation.find_boundary_coordinates(mask)
    snake_coords = active_contour(gaussian(255*mask, 3), boundary, alpha=snake_alpha, beta=snake_beta)
    return utils.segmentation.coord_array_to_contour_list(snake_coords)


def estimate_midline(mask):
    largest_object_boundary = utils.segmentation.extract_largest_object(mask, boundary_only=True)
    skeleton, distance = medial_axis(mask, return_distance=True)
    midline = calculate_midline(distance, largest_object_boundary)
    midline *= mask
    return midline, distance, skeleton


def process_masks_wrapper(params):
    frame_id, masks, annot_params = params
    assert isinstance(annot_params, AnnotParams)
    return frame_id, process_masks(masks, params=annot_params)


@dataclass
class AnnotParams:
    """
    verbose : bool, optional
        Whether to print out debug information about the progress, by default True
    clean : bool, optional
        Whether to clean the mask at the beginning, by default False
    convexity_threshold : float, optional
        If the convexity of the mask is smaller than this threshold, it will be dropped, by default 0.7
    smooth_degree : int, optional
        Degree of B-spline to smooth the mask, by default 4
    smooth_iter : int, optional
        Number od smoothing iterations, by default 1
    use_snakes : bool, optional
        Whether to use active contours instead of B-spline smoothing, by default False
    snake_alpha : float, optional
        Alpha parameter of the active contours method, by default 0.01
    snake_beta : float, optional
        Beta parameter of the active contours method, by default 0.3
    max_step_cost : int, optional
        Cost of a step that goes to point which is not on the midline, by default 3
    body_tail_ratio : float, optional
        If the body-tail ration of the actual rat is smaller than this threshold, the mask will be dropped, by default 0.9
    use_median : bool, optional
        Whether to use median or mean to determine the tail base point, by default True
    use_snake_skel : bool, optional
        Whether to use the midline of the original or the smoothed mask to determine the tail base point, by default True
    snd_ws_compactness : float, optional
        Compactness value for segmenting the mask into body parts, by default 1.0
    keep_orig_tail : bool, optional
        Whether to keep the original tail segment during body part segmentation, by default True
    use_head_corner : bool, optional
        Whether to use corner points for the head keypoint, by default True
    split_head_region : bool, optional
        Whether to split the head region, by default True
    """
    verbose: bool = True
    clean: bool = True
    convexity_threshold: float = 0.7
    smooth_degree: int = 4
    smooth_iter: int = 1
    use_snakes: bool = False
    snake_alpha: float = 0.01
    snake_beta : float = 0.3
    max_step_cost: int = 3
    body_tail_ratio: float = 0.7  # 0.9
    use_median: bool = True
    use_snake_skel: bool = True
    snd_ws_compactness: float = 1.0
    keep_orig_tail: bool = True
    use_head_corner: bool = False
    split_head_region: bool = True


def process_image(fn, i, vis_dir=None, params=None):
    assert os.path.isfile(fn) and os.path.exists(fn), f'{fn} does not exist'
    vis_fn = os.path.join(vis_dir, str(i) + '.png') if vis_dir is not None else None

    mask = cv2.imread(fn, -1)
    mask_ids = np.unique(mask)[1:]
    if len(mask_ids) == 0:
        return {}, np.zeros_like(mask, dtype=np.uint8), np.zeros_like(mask, dtype=np.uint8), i
    masks = [mask == id for id in mask_ids]

    if params is None:
        params = AnnotParams()
    else:
        assert isinstance(params, AnnotParams)
    keypoints, parts, instances = process_masks(masks, vis_fn=vis_fn, params=params)
    return keypoints, parts, instances, i


def process_image_wrapper(params):
    return process_image(*params)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mask_dir_base', required=True, help='Mask directory.')
    parser.add_argument('-o', '--out_dir_base', required=True, help='Output directory.')
    parser.add_argument('-v', '--vis_dir_base', help='Directory to save figures during calculation.')
    parser.add_argument('--overwrite', default=False, action='store_true', help='Overwrite existing results.')
    parser.add_argument('-p', '--params_json', help='Annotation parameters given by a json.')
    parser.add_argument('--no_seq_folder', default=False, action='store_true',
                        help='Whether the mask directory contains seq folders or not.')
    return parser.parse_args()


def run(mask_dir_base, out_dir_base, vis_dir_base=None, overwrite=False, params_json=None, no_seq_folder=False):
    if overwrite:
        if os.path.exists(out_dir_base):
            shutil.rmtree(out_dir_base)
        if vis_dir_base is not None and os.path.exists(vis_dir_base):
            shutil.rmtree(vis_dir_base)
    pool = multiprocessing.Pool()
    h, w = int(1.5*420), int(1.5*640)
    seqs = [''] if no_seq_folder else sorted([d for d in os.listdir(mask_dir_base) if d.find('aug_') == -1])

    if params_json is not None:
        params = AnnotParams(**utils.io.read(params_json))
    else:
        params = None

    for seq in seqs:
        mask_dir = os.path.join(mask_dir_base, seq).rstrip('/')
        out_dir = os.path.join(out_dir_base, seq).rstrip('/')
        utils.io.make_directory(out_dir)
        vis_dir = None
        if vis_dir_base is not None:
            vis_dir = os.path.join(vis_dir_base, seq).rstrip('/')
            utils.io.make_directory(vis_dir)

        # Calculate keypoints
        fns = utils.io.list_directory(mask_dir, sort_key=utils.io.filename_to_number)
        print(seq, len(fns))
        exists = not overwrite and os.path.exists(out_dir + '.json')
        if not exists:
            all_keypoints = {}
            for keypoints, parts, instances, i in pool.imap_unordered(process_image_wrapper,
                                                                      zip(fns,
                                                                          range(len(fns)),
                                                                          itertools.repeat(vis_dir),
                                                                          itertools.repeat(params))):
                all_keypoints[i] = keypoints
                print(i, keypoints)
                print(i, parts.shape, instances.shape)
                cv2.imwrite(os.path.join(out_dir, str(i) + '_parts.png'), parts)
                cv2.imwrite(os.path.join(out_dir, str(i) + '_instances.png'), instances)
                print(i)

            # Save keypoints
            utils.io.save(out_dir + '.json', all_keypoints)
        else:
            print('Already exists, skip keypoint generation')

        # Visualize
        if vis_dir_base is not None and not exists:
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            out_fn = os.path.join(vis_dir + '.avi')
            vid_out = cv2.VideoWriter(out_fn, fourcc, 10.0, (3*w, h))

            for i in range(len(fns)):
                fn = os.path.join(mask_dir, str(i) + '.png')
                orig = cv2.imread(fn, cv2.IMREAD_COLOR)
                orig = cv2.resize(orig, (w, h))
                mask0 = np.zeros_like(orig)
                out_fn0 = os.path.join(vis_dir, str(i) + '_0.png')
                if os.path.exists(out_fn0):
                    mask0 = cv2.imread(out_fn0)[10:-11, 10:-11]
                    mask0 = cv2.resize(mask0, (w, h))
                mask1 = np.zeros_like(orig)
                out_fn1 = os.path.join(vis_dir, str(i) + '_1.png')
                if os.path.exists(out_fn1):
                    mask1 = cv2.imread(out_fn1)[10:-11, 10:-11]
                    mask1 = cv2.resize(mask1, (w, h))
                frame = np.concatenate((orig, mask0, mask1), axis=1)
                vid_out.write(frame)
            vid_out.release()


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
