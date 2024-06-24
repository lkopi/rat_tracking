#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np

from utils.segmentation import single2multichannel


def generate_random_color():
    """Generates a random color.

    Returns
    -------
    list of int
        A randomly generated color.
    """
    return list(np.random.choice(range(256), size=3))


def generate_random_colors(n_colors, sorted=False):
    """Randomly generates a list of unique colors.

    Parameters
    ----------
    n_colors : int
        Number of unique colors to generate.

    Returns
    -------
    list of list of int
        A sorted list of randomly generated unique colors.
    """
    colors = np.unique([generate_random_color() for _ in range(n_colors)], axis=0)
    while len(colors) < n_colors:
        colors = np.concatenate((colors,
                                 np.asarray([generate_random_color()
                                             for _ in range(n_colors - len(colors))])), axis=0)
        colors = np.unique(colors, axis=0)
    if not sorted:
        rng = np.random.default_rng()
        rng.shuffle(colors, axis=0)
    return colors.tolist()


def map_labels(labels, label_map, dtype=None):
    """Map the given label matrix by the specified mapping dictionary.

    Parameters
    ----------
    labels : np.ndarray
        Label matrix consisting of hashable objects.
    label_map : dict of {hashable: object}
        Dictionary defining the mapping of the labels.
    dtype : data-type, optional
        Data type of the mapping target, if not specified use the type of the first value of `label_map`,
        by default None.

    Returns
    -------
    np.ndarray
        Mapped `labels` by the `label_map`.
    """
    if 0 == len(label_map):
        return labels.reshape((*labels.shape, 1))

    # assert type(label_map.keys()[0]) <= labels.dtype
    label_ids = np.unique(labels)  # sorted(label_map.keys())
    tmp = np.asarray(list(label_map.values())[0])

    dim = tmp.shape
    assert len(dim) <= 1
    if len(dim) == 0:
        dim = 1
    else:
        dim = dim[0]
    assert 0 < dim

    if dtype is None:
        dtype = tmp.dtype

    lookup_table = np.empty((label_ids[-1] + 1, dim), dtype=dtype)
    for label_id in label_ids:
        if label_id in label_map:
            lookup_table[label_id] = label_map[label_id]
        else:
            lookup_table[label_id] = label_id
    relabeled = lookup_table[labels.reshape(-1), :].reshape((*labels.shape, dim))
    return relabeled


def visualize_superpixels(superpixels, color_map=None):
    if color_map is None:
        label_ids = np.unique(superpixels)
        color_map = dict(zip(label_ids, generate_random_colors(len(label_ids))))
    images = map_labels(superpixels, color_map, dtype=np.uint8)
    return images


def is_grayscale(img):
    return img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1)


def grayscale_to_rgb(grayscale, make_copy=False):
    assert grayscale.ndim == 2 or grayscale.ndim == 3 or (grayscale.ndim == 4 and grayscale.shape[-1] == 1)
    if grayscale.ndim == 2 or (grayscale.ndim == 3 and 1 < grayscale.shape[-1]):
        grayscale = grayscale[..., None]
    if grayscale.ndim == 3 and grayscale.shape[-1] == 1:
        h, w = grayscale.shape[:2]
        out = np.broadcast_to(grayscale, (h, w, 3))
    else:
        c, h, w = grayscale.shape[:3]
        out = np.broadcast_to(grayscale, (c, h, w, 3))
    if make_copy:
        out = np.copy(out)
    return out


def concat_images(images, n_rows=1):
    assert 0 < len(images)
    if len(images) == 1:
        return images[0]

    n_cols = len(images) // n_rows
    assert n_cols * n_rows == len(images)
    concat_list = []
    i = 0
    for r in range(n_rows):
        temp = []
        for c in range(n_cols):
            temp.append([grayscale_to_rgb(images[i]) if is_grayscale(images[i]) else images[i]])
            i += 1
        concat_list.append(temp)
    return np.block(concat_list)


def overlay_mask(images, masks, alpha=0.7, color=(0, 0, 255)):
    assert images.dtype == np.uint8
    assert len(color) == 3

    colored = (grayscale_to_rgb(masks > 0) * color).astype(np.uint8)
    colored = cv2.addWeighted(images, alpha, colored, 1 - alpha, 0)
    out = images.copy()
    out[masks > 0] = colored[masks > 0]
    return out


def visualize_predictions(predictions):
    if predictions.ndim == 3 and 3 < predictions.shape[-1]:
        predictions = single2multichannel(predictions)
    assert predictions.ndim == 4

    predictions = (255 * (predictions > 0)).astype(np.uint8)
    l, h, w, ch = predictions.shape
    if ch < 3:
        tmp = np.zeros((l, h, w, 3), dtype=np.uint8)
        tmp[..., :ch] = predictions
        predictions = tmp
    else:
        predictions = predictions[..., :3]
    return predictions