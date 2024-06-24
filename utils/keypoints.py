#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np


def calculate_weighted_center(mask, round_coords=True):
    assert mask.ndim == 2 or (mask.ndim == 3 and mask.shape[-1] == 1)
    # Create a grid shaped like the image
    x = range(0, mask.shape[1])
    y = range(0, mask.shape[0])
    (X,Y) = np.meshgrid(x,y)

    # Calculate the weighted center
    x_coord = (X*mask).sum() / (mask.sum() + 0.00001)
    y_coord = (Y*mask).sum() / (mask.sum() + 0.00001)
    if round_coords:
        return int(round(x_coord)), int(round(y_coord))
    return x_coord, y_coord


def heatmap2keypoints(htmp, round_coords=True):
    assert htmp.ndim == 3
    kps = []
    for idx in range(htmp.shape[-1]):
        try:
            kps.append(list(calculate_weighted_center(htmp[..., idx],
                                                      round_coords=round_coords)[::-1]))
        except:
            kps.append([])
    return kps


def keypoints2heatmap(landmarks, shape, radius=2, thickness=2, apply_gauss=True):
    def _keypoints2heatmap(_landmarks):
        h, w = shape[:2]
        htmp = np.zeros((h, w, 3), dtype=np.uint8)
        for i, l in enumerate(_landmarks):
            if l == []:
                continue
            c = [0] * 3
            c[i] = 255
            cv2.circle(htmp, center=tuple(l[:2])[::-1], radius=radius, color=c, thickness=thickness)

        if apply_gauss:
            kernel_size = radius + thickness + 1
            htmp = cv2.GaussianBlur(htmp, (kernel_size, kernel_size), 0)

        return htmp

    htmps = []
    for i in range(0, len(landmarks), 3):
        _landmarks = landmarks[3*i:3*(i + 1)]
        if len(_landmarks) < 3:
            _landmarks = (_landmarks*3)[:3]
        htmps.append(_keypoints2heatmap(_landmarks))

    if len(htmps) == 1:
        htmp = htmps[0]
    else:
        htmp = np.stack(htmps, axis=-1)

    return htmp