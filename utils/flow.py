#!/usr/bin/env python
# coding: utf-8

import math
import os
import sys

import cv2
import numpy as np


def read_flow(filename):
    assert type(filename) is str, "file is not str %r" % str(filename)
    assert os.path.isfile(filename) is True, "file does not exist %r" % str(filename)
    assert filename[-4:] == '.flo', "file ending is not .flo %r" % filename[-4:]
    f = open(filename, 'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == 202021.25, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)

    data = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()

    return flow


def get_average_flow(flow, channel_wise_segments, min_area=100):
    avg = np.zeros_like(flow)
    for ch in range(channel_wise_segments.shape[-1]):
        mask = (channel_wise_segments[..., ch] > 0)
        masked_data = flow[mask]
        if min_area < len(masked_data != 0):
            mean_flow = np.mean(masked_data[masked_data != 0])
            if not np.isnan(mean_flow):
                avg[mask] = mean_flow
    return avg


def create_grid(width, height):
    x_mat = (np.expand_dims(range(width), 0) * np.ones((height, 1), dtype=np.int32)).astype(np.int32)
    y_mat = (np.ones((1, width), dtype=np.int32) * np.expand_dims(range(height), 1)).astype(np.int32)
    return x_mat, y_mat


def _convert_displacement(displacement, relative_to_absolute=True, swap_uv_channels=False, clip=True):
    """
    By default, the channels should represent the displacement along the rows (y axis) and the columns (x axis),
    respectively. If not, set the `swap_uv_channels` to True.
    """
    height, width, _ = displacement.shape
    x_mat, y_mat = create_grid(width, height)
    if swap_uv_channels:
        displacement = displacement[..., ::-1]

    displacement = displacement.copy()
    if relative_to_absolute:
        displacement[..., 0] += y_mat
        displacement[..., 1] += x_mat
    else:
        displacement[..., 0] -= y_mat
        displacement[..., 1] -= x_mat

    if clip:
        displacement[..., 0] = np.clip(displacement[..., 0], 0, height - 1)
        displacement[..., 1] = np.clip(displacement[..., 1], 0, width - 1)

    return displacement


def relative_to_absolute(displacement, swap_uv_channels=False, clip=True):
    return _convert_displacement(displacement, relative_to_absolute=True, swap_uv_channels=swap_uv_channels, clip=clip)


def absolute_to_relative(displacement, swap_uv_channels=False, clip=True):
    return _convert_displacement(displacement, relative_to_absolute=False, swap_uv_channels=swap_uv_channels, clip=clip)


def warp_image(image, displacement, is_relative=True, swap_uv_channels=False):
    assert image.shape[:2] == displacement.shape[:2]

    displacement = displacement.astype(np.int32)
    if swap_uv_channels:
        displacement = displacement[..., ::-1]
    if is_relative:
        displacement = relative_to_absolute(displacement)

    warped_image = image.copy()
    warped_image[displacement[..., 0], displacement[..., 1]] = image
    return warped_image


def draw_flow_vectors(img, disparities, max_spin_size=5.0, spacing=20, is_relative=True, color=(0, 255, 0),
                      swap_uv_channels=False):
    """
    By default, the channels should represent the displacement along the rows (y axis) and the columns (x axis),
    respectively. If not, set the `swap_uv_channels` to True.
    """
    assert disparities.ndim == 3 and img.ndim == 3
    assert disparities.shape[:2] == img.shape[:2]
    assert disparities.shape[2] == 2 and img.shape[2] == 3

    if swap_uv_channels:
        disparities = disparities[..., ::-1]

    height, width, _ = disparities.shape
    img = np.copy(img)  # height x width x channels
    disparities = disparities.astype(np.int32)
    if not is_relative:
        disparities = absolute_to_relative(disparities)

    # Compute the maximum length (longest flow)
    max_length = 0.
    for x in range(0, height, spacing):
        for y in range(0, width, spacing):
            cx, cy = x, y
            dx, dy = disparities[cx, cy]
            length = math.sqrt(dx * dx + dy * dy)
            if length > max_length:
                max_length = length

    # Draw arrows
    for x in range(0, height, spacing):
        for y in range(0, width, spacing):
            cx, cy = x, y
            dx, dy = disparities[cx, cy]
            length = math.sqrt(dx * dx + dy * dy)

            if length > 0:
                # Factor to normalise the size of the spin depending on the length of the arrow
                spin_size = max_spin_size * length / max_length
                nx, ny = int(cx + dx), int(cy + dy)

                nx = min(max(0, nx), height - 1)
                ny = min(max(0, ny), width - 1)

                cx, cy = cy, cx
                nx, ny = ny, nx

                cv2.line(img, (cx, cy), (nx, ny), color, 1, cv2.LINE_AA)

                # Draws the spin of the arrow
                angle = math.atan2(cy - ny, cx - nx)

                cx = int(nx + spin_size * math.cos(angle + math.pi / 4))
                cy = int(ny + spin_size * math.sin(angle + math.pi / 4))
                cv2.line(img, (cx, cy), (nx, ny), color, 1, cv2.LINE_AA, 0)

                cx = int(nx + spin_size * math.cos(angle - math.pi / 4))
                cy = int(ny + spin_size * math.sin(angle - math.pi / 4))
                cv2.line(img, (cx, cy), (nx, ny), color, 1, cv2.LINE_AA, 0)
    return img


def visualize_flow_vectors(images, flows, spacing=10, swap_uv_channels=False):
    """
    By default, the channels should represent the displacement along the rows (y axis) and the columns (x axis),
    respectively. If not, set the `swap_uv_channels` to True.
    """
    l, h, w = images.shape[:3]
    # fl, fh, fw = flows.shape[:3]
    # dh, dw = (h - fh) // 2, (w - fw) // 2
    outs = np.empty((l, h, w, 3), dtype=np.uint8)
    for i, flow in enumerate(flows):
        outs[i] = images[i].copy()
        tmp = images[i]
        outs[i] = draw_flow_vectors(tmp, flow, spacing=spacing, swap_uv_channels=swap_uv_channels)
    return outs


def _make_color_wheel():
    #  color encoding scheme

    #   adapted from the color circle idea described at
    #   http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    n_cols = RY + YG + GC + CB + BM + MR
    color_wheel = np.zeros([n_cols, 3])  # r g b

    col = 0
    # RY
    color_wheel[0:RY, 0] = 255
    color_wheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY

    # YG
    color_wheel[col:YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    color_wheel[col:YG + col, 1] = 255
    col += YG

    # GC
    color_wheel[col:GC + col, 1] = 255
    color_wheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC

    # CB
    color_wheel[col:CB + col, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    color_wheel[col:CB + col, 2] = 255
    col += CB

    # BM
    color_wheel[col:BM + col, 2] = 255
    color_wheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM

    # MR
    color_wheel[col:MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    color_wheel[col:MR + col, 0] = 255

    return color_wheel


def _compute_color(u, v):
    color_wheel = _make_color_wheel()
    nan_u = np.where(np.isnan(u))
    nan_v = np.where(np.isnan(v))

    u[nan_u], u[nan_v] = 0, 0
    v[nan_u], v[nan_v] = 0, 0

    n_cols = color_wheel.shape[0]
    radius = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (n_cols - 1)  # -1~1 mapped to 1~n_cols
    k0 = fk.astype(np.uint8)  # 1, 2, ..., n_cols
    k1 = k0 + 1
    k1[k1 == n_cols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1], 3])
    n_colors = color_wheel.shape[1]
    for i in range(n_colors):
        tmp = color_wheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75  # out of range
        img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8)

    return img.astype(np.uint8)


def _compute_img(flow, verbose=False):
    eps = sys.float_info.epsilon

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # fix unknown flow
    greater_u = np.where(u > 1e9)
    greater_v = np.where(v > 1e10)
    u[greater_u], u[greater_v] = 0, 0
    v[greater_u], v[greater_v] = 0, 0

    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    max_rad = max([-1, np.amax(rad)])
    if verbose:
        min_u = min([999, np.amin(u)])
        max_u = max([-999, np.amax(u)])
        min_v = min([999, np.amin(v)])
        max_v = max([-999, np.amax(v)])
        print('max flow: %.4f flow range: u = %.3f .. %.3f v = %.3f .. %.3f\n' % (max_rad, min_u, max_u, min_v, max_v))

    u = u / (max_rad + eps)
    v = v / (max_rad + eps)
    img = _compute_color(u, v)
    return img


def visualize_flows(flows, h, w):
    fl, fh, fw = flows.shape[:3]
    images = np.empty((fl, h, w, 3), dtype=np.uint8)
    for i, flow in enumerate(flows):
        images[i] = cv2.resize(_compute_img(flow), (w, h))
    return images
