#!/usr/bin/env python
# coding: utf-8

from functools import partial
import math
import multiprocessing

import numpy as np
from scipy import ndimage as ndimage
from skimage.draw import polygon
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
from skimage.morphology import closing, dilation, erosion, opening, disk, square, convex_hull_image
from tqdm import tqdm


def coord_array_to_contour_list(coords, flip=True):
    """Converts the given (N, 2) (y and x) coordinate array to a contour list.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2) array of boundary coordinates. Each row contains y (row) and x (column) coordinates in this order.
    flip: bool
        If true, then it flips the x and y coordinates

    Returns
    -------
    list of float
        List of x and y contour coordinates. e.g. [x1, y1, x2, y2, ..., xN, yN].
    """
    assert coords.ndim == 2
    assert 0 < coords.shape[0] and coords.shape[1] == 2
    if flip:
        coords = coords[:, ::-1]
    return coords.ravel().tolist()


def coord_lists_to_coord_array(xlist, ylist):
    assert len(xlist) == len(ylist)
    return np.asarray(list(zip(xlist, ylist)))


def coord_lists_to_contour_list(xlist, ylist):
    """Converts the given x-coordinate list and y-coordinate list to a contour list.

    Parameters
    ----------
    xlist : list of float
        List of x (column) coordinates.
    ylist : list of float
        List of y (row) coordinates.

    Returns
    -------
    list of float
        List of x and y contour coordinates. e.g. [x1, y1, x2, y2, ..., xN, yN].
    """
    return coord_array_to_contour_list(coord_lists_to_coord_array(xlist, ylist), flip=False)


def contour_list_to_coord_array(mask_contour):
    assert len(mask_contour) % 2 == 0
    coords = np.empty((len(mask_contour) // 2, 2), np.float32)
    coords[:, 0], coords[:, 1] = mask_contour[::2], mask_contour[1::2]
    return coords


def apply_morphological_operation(labels, operation='closing', ktype='disk', ksize=6):
    # Examples: https://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html
    if operation == 'closing':
        operation = closing
    elif operation == 'dilation':
        operation = dilation
    elif operation == 'erosion':
        operation = erosion
    elif operation == 'opening':
        operation = opening
    else:
        raise NotImplementedError()

    if ktype == 'disk':
        kernel = disk
    elif ktype == 'square':
        kernel = square
    else:
        raise NotImplementedError()

    out_mask = np.zeros_like(labels)
    label_ids = np.unique(labels)[1:]
    selem = kernel(ksize)

    for label_id in label_ids:
        mask = (labels == label_id)
        mask = operation(mask, selem)
        out_mask[mask] = label_id
    return out_mask


def apply_closing(labels, ksize=6, ktype='disk'):
    return apply_morphological_operation(labels, operation='closing', ktype=ktype, ksize=ksize)


def apply_opening(labels, ksize=6, ktype='square'):
    return apply_morphological_operation(labels, operation='opening', ktype=ktype, ksize=ksize)


def is_convex(mask, threshold=0.8, return_overlap=False):
    """Returns true if the object represented by the given mask is convex.

    Parameters
    ----------
    mask : np.ndarray
        2-dimensional binary mask.
    threshold : float, optional
        Convexity threshold, by default 0.8
    return_overlap : bool, optional
        If true, then it also returns the overlap between the original mask and its convex hull

    Returns
    -------
    bool
        True if the object represented by the given mask is convex.
    """
    chull = convex_hull_image(mask)
    dsum = float(np.sum(mask))
    hsum = float(np.sum(chull))
    overlap = dsum/hsum
    if return_overlap:
        return overlap > threshold, overlap
    return overlap > threshold


def find_boundary_coordinates(mask):
    """Returns the boundary of the objects represented by the mask.

    Parameters
    ----------
    mask : np.ndarray
        2-dimensional binary mask.

    Returns
    -------
    np.ndarray
        List of N x 2 float arrays, where each row represents a coordinate of the boundary of an object.
    """
    contours = sorted(find_contours(mask, 0.5), key=lambda x: len(x), reverse=True)
    return contours


def coordinates_to_mask(coordinates, shape):
    """Convert a 2-dimensional index-array to a binary mask.

    Parameters
    ----------
    coordinates : np.ndarray
        Nx2 array of coordinates.
    shape : (int, int)
        Shape of the mask.

    Returns
    -------
    np.ndarray
        2-dimensional binary mask.
    """
    mask = np.zeros(shape, dtype='bool')
    mask[np.round(coordinates[:, 0]).astype('int'), np.round(coordinates[:, 1]).astype('int')] = 1
    return mask


def boundary_to_mask(boundary, shape, fill=False):
    """Convert a boundary array to a binary mask.

    Parameters
    ----------
    boundary : np.ndarray
        Nx2 array of boundary coordinates.
    shape : (int, int)
        Shape of the mask.
    fill : bool, optional
        Either to fill in the mask, by default False

    Returns
    -------
    np.ndarray
        2-dimensional binary boundary/object mask.
    """
    # assert 0 < len(boundary)
    if isinstance(boundary, list):
        mask = np.zeros(shape, dtype=np.bool)
        for b in boundary:
            mask = np.logical_or(mask, boundary_to_mask(b, shape))
        return mask

    mask = coordinates_to_mask(boundary, shape)
    if fill:
        mask = ndimage.binary_fill_holes(mask)
    return mask


def mask_border(mask, size=1, value=0):
    mask = mask.copy()
    if size < 1:
        return mask

    border_mask = np.ones_like(mask, dtype=np.bool)
    border_mask[size:-size, size:-size] = 0
    mask[border_mask] = value
    return mask


def extract_largest_object(mask, boundary_only=False):
    """
    Parameters
    ----------
    mask : np.ndarray
        2-dimensional binary mask.
    boundary_only : bool, optional
        If true, then it only returns the boundary og the largest object
    """
    assert mask.ndim == 2
    mask = mask_border(mask)  # Make that the mask doesn't connected to the border
    return boundary_to_mask(find_boundary_coordinates(mask)[0], shape=mask.shape, fill=not boundary_only)


def encode_mask(mask, tolerance=1.5):
    """Encode the given mask into a contour list that contains x and y coordinates approximated by some polygons.
    The resolution can be controlled by the tolerance.

    Parameters
    ----------
    mask : np.ndarray
        Binary 2D mask.
    tolerance: float
        Controls the maximum distance from original boundary points and the approximated boundary points, default 1.5.

    Returns
    -------
    list of list of float
        List of encoded masks, given by contour lists. e.g. [[x1, y1, x2, y2, ..., xN, yN], ...].
    """
    contours = find_boundary_coordinates(mask)
    segmentations = []
    for contour in contours:
        contour = approximate_polygon(contour, tolerance=tolerance)
        segmentation = coord_array_to_contour_list(contour)
        if len(segmentation) % 2 == 0 and len(segmentation) >= 6:
            segmentations.append(segmentation)
    return segmentations


def decode_mask(mask_contour, img_shape):
    """Recover the original mask from the (encoded) contour list. (List of contours can also be given.)

    Parameters
    ----------
    mask_contour : list of list of float OR list of float
        Elements ar list of x and y contour coordinates. e.g. [x1, y1, x2, y2, ..., xN, yN].
        x represents the column and y represents the row of the image.
        If a list of list is given, then it decodes and merge each of them.
    img_shape : tuple of int
        Height and width of the original image.

    Returns
    -------
    np.ndarray
        Recovered binary mask from the given contour.
    """
    assert 0 < len(mask_contour)
    if isinstance(mask_contour[0], list):
        mask = np.zeros(img_shape, dtype=np.bool)
        for mc in mask_contour:
            mask = np.logical_or(mask, decode_mask(mc, img_shape))
        return mask

    fill_row_coords, fill_col_coords = polygon(mask_contour[1::2], mask_contour[0::2], img_shape)
    mask = np.zeros(img_shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def smooth_mask(mask, n_iter=1, degree=2):
    """Iteratively smooth the given mask with B-splines.

    Parameters
    ----------
    mask : np.ndarray
        2D binary mask.
    n_iter : int, optional
        Number of iterations, by default 1.
    degree : int, optional
        Degree of the B-spline, by default 2.

    Returns
    -------
    list of np.ndarray
        List of approximated contours.
    """
    out = []
    for encoded in encode_mask(mask):
        decoded = np.array(list(zip(encoded[1::2], encoded[0::2])))
        for _ in range(n_iter):
            decoded = subdivide_polygon(decoded, degree=degree, preserve_ends=True)
        # out.append(decoded)
        out.append(coord_array_to_contour_list(decoded))
    return out


def crop_center(img, cropped_height=None, cropped_width=None, channel_first=False):
    assert 2 <= img.ndim
    if channel_first:
        img_t = np.transpose(img, (*range(1, img.ndim), 0))
        res = crop_center(img_t, cropped_height=cropped_height, cropped_width=cropped_width, channel_first=False)
        return np.transpose(res, (img.ndim-1, *range(0, img.ndim-1)))
    h, w = img.shape[:2]
    if cropped_height is None:
        cropped_height = h
    if cropped_width is None:
        cropped_width = w

    assert cropped_height % 2 == 0 and cropped_width % 2 == 0
    assert cropped_height <= h and cropped_width <= w

    hc = (h - cropped_height) // 2
    wc = (w - cropped_width) // 2
    if 0 < hc:
        img = img[hc:-hc, :]
    if 0 < wc:
        img = img[:, wc:-wc]
    return img


def foreground_proposal(labels, gt_masks, threshold=0.5):
    ious, label_ids = mask_intersection_over_union(labels, gt_masks, use_iou=False)

    fg_indices = np.where(ious > threshold)
    fg_ids = label_ids[fg_indices]
    fg_masks = np.isin(labels, fg_ids)

    return fg_masks, fg_ids


def mask_intersection_over_union(mask_a, mask_b, use_iou=True, take_smaller=True):
    """
    mask_a: multi-label mask
    mask_b: binary mask
    """
    label_ids, intersect_counts = np.unique(mask_a[mask_b > 0], return_counts=True)
    _, all_counts = np.unique(mask_a[np.isin(mask_a, label_ids)], return_counts=True)
    if use_iou:
        ious = intersect_counts / (all_counts + np.count_nonzero(mask_b) - intersect_counts)
    else:
        ious1 = intersect_counts / all_counts
        ious2 = intersect_counts / np.count_nonzero(mask_b)
        ious = np.minimum(ious1, ious2) if take_smaller else np.maximum(ious1, ious2)
    if len(label_ids) == 0:
        return np.asarray([0.]), np.asarray([0])
    elif label_ids[0] == 0:
        if 1 < len(label_ids):
            return ious[1:], label_ids[1:]
        else:
            return np.asarray([0.]), np.asarray([0])
    else:
        return ious, label_ids


def is_multichannel(masks):
    return masks.ndim == 4


def single2multichannel(masks):
    values = np.unique(masks)[1:]
    out = np.zeros((*masks.shape, len(values)), dtype=masks.dtype)
    for ch, value in enumerate(values):
        tmp = (masks == value)
        out[tmp, ch] = ch + 1
    return out


def multi2singlechannel(masks):
    ch = masks.shape[-1]
    is_binary = len(np.unique(masks)) == 2
    out = np.zeros(masks.shape[:-1], dtype=masks.dtype)
    for c in range(0, ch):
        tmp = (masks[..., c] > 0)
        out[tmp] = c + 1 if is_binary else masks[tmp, c]
    return out


def apply_transformation(images, func):
    vfunc = np.vectorize(func)
    return vfunc(images)


def apply_transformation_parallel(images, func, out_shape=None, out_dtype=None, chunksize=100, n_processes=-1):
    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, max(1, len(images) // chunksize))
    if n_processes == 1:
        return apply_transformation(images, func)

    if out_shape is None:
        out_shape = images.shape
    if out_dtype is None:
        out_dtype = images.dtype

    transformed_images = np.empty(out_shape, dtype=out_dtype)
    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm(total=int(math.ceil(len(images)/n_processes))) as pbar:
            for i, out in enumerate(pool.imap(partial(apply_transformation, func=func), images)):
                transformed_images[i*chunksize:(i+1)*chunksize] = out
                pbar.update()

    return transformed_images
