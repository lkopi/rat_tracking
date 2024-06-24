#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import utils.segmentation


class PlotLogger:
    def __init__(self, shape):
        self.fig, self.ax = plt.subplots(figsize=(18, 12))
        self.shape = shape
        self.reset()

    def plot(self, message=None, image=None, coordinate_list=None, colors=None, size=2, plot_as_dots=False):
        if coordinate_list is not None and colors is not None:
            assert len(coordinate_list) <= len(colors), "Coordinate list cannot be longer than colors"

        if image is not None:
            self.ax.imshow(image, zorder=0)

        if coordinate_list is not None:
            for idx, coords in enumerate(coordinate_list):
                if isinstance(coords, list):
                    coords = utils.segmentation.contour_list_to_coord_array(coords)
                color = colors[idx] if colors is not None else None
                if not plot_as_dots:
                    self.ax.plot(coords[:, 1], coords[:, 0], color=color, lw=size)
                else:
                    self.ax.plot(coords[:, 1], coords[:, 0], color=color, marker='o', linestyle='None', markersize=size)

        if message is not None:
            self.ax.text(20, 25, message, fontsize=16)

    def reset(self):
        self.ax.clear()
        self.ax.imshow(np.zeros(self.shape, dtype=np.uint8), zorder=0)
        self.ax.set_xticks([]), self.ax.set_yticks([])
        self.ax.axis([0, self.shape[1], self.shape[0], 0])
        self.ax.axis('off')

    def save_figure(self, fn, reset_figure=True):
        self.fig.savefig(fn, bbox_inches='tight')
        if reset_figure:
            self.reset()

    def __del__(self):
        plt.close(self.fig)
