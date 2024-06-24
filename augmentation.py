#!/usr/bin/env python
# coding: utf-8

import sys
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import scipy.ndimage as ndimage
from skimage.morphology import dilation, square
from skimage.transform import rotate, rescale

WBAUG_PATH = './external/WB_color_augmenter/WBAugmenter_Python/'
sys.path.append(WBAUG_PATH)
from WBAugmenter import WBEmulator as wbAug

from image_warp import warp_images
from utils.keypoints import heatmap2keypoints
from utils.io import copy_directory


@dataclass
class BlurType(Enum):
    none = None
    gauss = 'gauss'
    bilateral = 'bilateral'
    median = 'median'
    edge = 'edge'


@dataclass
class AugmentParams:
    warping: bool = False
    distortion: int = 10
    shift: tuple = (0, 0)
    shift_upper: bool = True
    blur_shifted: bool = True
    blur_stationary: bool = False
    blur_type: BlurType = None
    rot_angle: float = 45.
    roi_width: tuple = (190, 430)  # (0, 640)
    roi_height: tuple = (10, 410)  # (0, 420)
    scale_shifted: float = 1.1
    apply_shadow: bool = False
    dilation_size: int = 10  # 15


def _mitigate_translation(shift, coords, min_coord=0, max_coord=np.inf):
    """Mitigate translation to not exceed the given boundary coordinates.

    shift: float
        Shift each coordinate with this value.
    coords: list of int
        List of coordinates in the shift direction.
    min_coord: int
        Minimum coordinate, by default 0
    max_coord: int
        Maximum coordinate, by default np.inf

    Returns
    -------
    float
        Mitigated shift value that won't exceed the boundary coordinates.
    """
    if max_coord <= shift + max(coords):
        d = min(0, max_coord - (shift + max(coords) + 1))
    elif shift + min(coords) < min_coord:
        d = max(0, min_coord - (shift + min(coords)))
    else:
        d = 0
    return d + shift


def _get_channel_ids(upper_id):
    """Calculate the chanel ids given the upper instance id.

    Parameters
    ----------
    upper_id : int
        Channel id of the upper instance.

    Returns
    -------
    (int, int, np.ndarray, np.ndarray)
        Channel id of the upper instance, channel id of the lower instance,
        keypoint ids of the upper instance, keypoint ids of the lower instance.
    """
    fst_id, snd_id = upper_id, 1 - upper_id
    fst_kp_channels = np.arange(3) + 3 * fst_id
    snd_kp_channels = np.arange(3) + 3 * snd_id
    return fst_id, snd_id, fst_kp_channels, snd_kp_channels


def _gather_upper_instance_coordinates(instances, upper_id, keypoints=None):
    fst_id, snd_id, fst_kp_channels, snd_kp_channels = _get_channel_ids(upper_id)

    whr0, whr1 = np.where(instances[..., fst_id])
    whr0kp, whr1kp = None, None
    if keypoints is not None:
        whr0kp, whr1kp = np.where(keypoints[..., fst_kp_channels].sum(axis=-1) > 0)
        whr0 = np.concatenate((whr0, whr0kp), axis=0)
        whr1 = np.concatenate((whr1, whr1kp), axis=0)

    return whr0, whr1, whr0kp, whr1kp


def _apply_translation(image, instances, upper_id, move,
                       bg=None, keypoints=None, parts=None,
                       orig_image=None, orig_instances=None, orig_keypoints=None, orig_parts=None,
                       roi_width=(190, 430), roi_height=(10, 410), dilation_size=15):
    """Shift the given mask by the specified `move` translation vector.

    Parameters
    ----------
    image : np.ndarray
        RGB image.
    instances : np.ndarray
        2-channel binary mask of the instances.
    move: (float, float)
        Translation vector.
    upper_id : int
        Channel number of the instance to shift.
    bg : np.ndarray, optional
        Background image, by default None.
    keypoints : np.ndarray, optional
        Keypoint heatmap, by default None.
    parts : np.ndarray, optional
        3-channel binary mask of the body-parts, by default None.
    roi_width : tuple, optional
        Work area, by default (190, 430).
    roi_height : tuple, optional
        Work area, by default (10, 410).

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
        Translated instances, image, keypoints and body-parts. None for values that was not specified.
    """
    # Mitigate translation
    whr0, whr1, whr0kp, whr1kp = _gather_upper_instance_coordinates(instances, upper_id, keypoints=keypoints)

    new_move = (_mitigate_translation(move[0], whr0, min_coord=roi_height[0], max_coord=roi_height[1]),
                _mitigate_translation(move[1], whr1, min_coord=roi_width[0], max_coord=roi_width[1]))

    whr0, whr1 = np.where(instances[..., upper_id])
    moved_whr0 = whr0 + new_move[0]
    moved_whr1 = whr1 + new_move[1]

    # Make sure that it stayed within the roi boundary
    assert max(moved_whr0) < roi_height[1], '({}, {})'.format(min(moved_whr0), max(moved_whr0))
    assert min(moved_whr0) >= roi_height[0], '({}, {})'.format(min(moved_whr0), max(moved_whr0))
    assert max(moved_whr1) < roi_width[1], '({}, {})'.format(min(moved_whr1), max(moved_whr1))
    assert min(moved_whr1) >= roi_width[0], '({}, {})'.format(min(moved_whr1), max(moved_whr1))

    # Get channel ids
    fst_id, snd_id, fst_kp_channels, snd_kp_channels = _get_channel_ids(upper_id)

    # Determine translated instances
    base_instances = orig_instances if orig_instances is not None else instances
    new_instances = np.zeros_like(base_instances)
    new_instances[..., snd_id] = base_instances[..., snd_id]
    new_instances[moved_whr0, moved_whr1, fst_id] = instances[whr0, whr1, fst_id]

    # Determine translated parts
    new_parts = None
    if parts is not None:
        base_parts = orig_parts if orig_parts is not None else parts
        new_parts = base_parts.copy()
        new_parts[..., fst_id] = 0
        new_parts[moved_whr0, moved_whr1, fst_id] = parts[whr0, whr1, fst_id]

    # Determine translated keypoints
    new_keypoints = None
    if keypoints is not None:
        base_keypoints = orig_keypoints if orig_keypoints is not None else keypoints
        new_keypoints = np.zeros_like(base_keypoints)
        new_keypoints[whr0kp + new_move[0], whr1kp + new_move[1]] = keypoints[whr0kp, whr1kp]
        new_keypoints[..., snd_kp_channels] = 0
        new_keypoints[..., snd_kp_channels] = base_keypoints[..., snd_kp_channels]

    # Inpaint translated instance
    if bg is None:
        bg = np.zeros_like(image)
    base_image = orig_image if orig_image is not None else image
    new_image = base_image.copy()
    dilated_instances = dilation(base_instances[..., fst_id] > 0, square(dilation_size))
    new_image[dilated_instances] = bg[dilated_instances]
    new_image[moved_whr0, moved_whr1, :] = image[whr0, whr1, :]

    return new_instances, new_image, new_keypoints, new_parts


def _apply_transform(image, instances, upper_id,
                     bg=None, keypoints=None, parts=None,
                     transform='rotate', rot_angle=0, scale_shifted=1.0,
                     roi_width=(190, 430), roi_height=(10, 410), dilation_size=15):
    """
    transform: str
    Either rotate or rescale
    """
    # Determine current location
    whr0, whr1, _, _ = _gather_upper_instance_coordinates(instances, upper_id, keypoints=keypoints)

    # Determine bbox and rotate it
    h1, w1, h2, w2 = max(0, min(whr0) - 5), max(0, min(whr1) - 5), min(instances.shape[0], max(whr0) + 6), min(
        instances.shape[1], max(whr1) + 6)
    new_keypoints, new_parts = None, None
    if transform == 'rotate':
        new_instances = rotate(instances[h1:h2, w1:w2], -rot_angle, resize=True,
                               preserve_range=True, order=0).astype(np.bool)
        if keypoints is not None:
            new_keypoints = rotate(keypoints[h1:h2, w1:w2], -rot_angle, resize=True,
                                   preserve_range=True, order=0).astype(np.uint8)
        if parts is not None:
            new_parts = rotate(parts[h1:h2, w1:w2], -rot_angle, resize=True,
                               preserve_range=True, order=0).astype(np.uint8)
        new_image = rotate(image[h1:h2, w1:w2], -rot_angle, resize=True, preserve_range=True).astype(np.uint8)
    elif transform == 'rescale':
        new_instances = rescale(instances[h1:h2, w1:w2], scale_shifted, anti_aliasing=False,
                                preserve_range=True, multichannel=True, order=0).astype(np.bool)
        if keypoints is not None:
            new_keypoints = rescale(keypoints[h1:h2, w1:w2], scale_shifted, anti_aliasing=False,
                                    preserve_range=True, multichannel=True, order=0).astype(np.uint8)
        if parts is not None:
            new_parts = rescale(parts[h1:h2, w1:w2], scale_shifted, anti_aliasing=False,
                                preserve_range=True, multichannel=True, order=0).astype(np.uint8)
        new_image = rescale(image[h1:h2, w1:w2], scale_shifted, anti_aliasing=True,
                            preserve_range=True, multichannel=True).astype(np.uint8)
    else:
        raise NotImplementedError

    # Determine location after rotation
    upper_center = ndimage.measurements.center_of_mass(instances[..., upper_id])
    move = (int(upper_center[0]) - (h2 - h1) // 2, int(upper_center[1])) - (w2 - w1) // 2
    new_instances, new_image, new_keypoints, new_parts = _apply_translation(new_image, new_instances,
                                                                            upper_id, move, bg=bg,
                                                                            keypoints=new_keypoints,
                                                                            parts=new_parts,
                                                                            orig_image=image,
                                                                            orig_instances=instances,
                                                                            orig_keypoints=keypoints,
                                                                            orig_parts=parts,
                                                                            roi_width=roi_width,
                                                                            roi_height=roi_height,
                                                                            dilation_size=dilation_size)

    return new_instances, new_image, new_keypoints, new_parts


def _apply_warping(image, instances, upper_id, keypoints,
                   bg=None, parts=None, distortion=15,
                   roi_width=(190, 430), roi_height=(10, 410), dilation_size=15):
    """Similar to apply_transform, but here we need to prepare more things."""
    fst_id, snd_id, fst_kp_channels, snd_kp_channels = _get_channel_ids(upper_id)

    # Get coordinates
    fst_kps = heatmap2keypoints(keypoints[..., fst_kp_channels])
    from_points, to_points = [], []
    for coord in fst_kps:
        from_points.append(coord)
        to_points.append([max(0, coord[0] + np.random.randint(-distortion, distortion + 1)),
                          max(0, coord[1] + np.random.randint(-distortion, distortion + 1))])

    # Define RoI
    whr0, whr1 = np.where(instances[..., fst_id])
    border = 30
    h1, w1, h2, w2 = max(0, min(whr0) - border), max(0, min(whr1) - border), max(whr0) + border + 1, max(
        whr1) + border + 1
    output_region = (h1, w1, h2, w2)

    # Apply thin-plat spline warping
    images = [instances[..., 0], instances[..., 1],
              image[..., 0], image[..., 1], image[..., 2]]
    for i in range(6):
        images.append(keypoints[..., i])
    if parts is not None:
        images.append(parts[..., 0])
        images.append(parts[..., 1])

    outs = warp_images(from_points, to_points, images, output_region, interpolation_order=0)

    # Gather results
    new_instances = np.stack(outs[:2], axis=2)
    new_instances[..., snd_id] = 0
    new_image = np.stack(outs[2:5], axis=2)
    new_keypoints = np.stack(outs[5:11], axis=2)
    new_parts = np.stack(outs[11:], axis=2) if parts is not None else None

    # Apply it on the whole image
    upper_center = ndimage.measurements.center_of_mass(instances[..., fst_id])  # ismétlődik
    move = (int(upper_center[0]) - (h2 - h1) // 2, int(upper_center[1])) - (w2 - w1) // 2
    new_instances, new_image, new_keypoints, new_parts = _apply_translation(new_image, new_instances,
                                                                            upper_id, move, bg=bg,
                                                                            keypoints=new_keypoints,
                                                                            parts=new_parts,
                                                                            orig_image=image,
                                                                            orig_instances=instances,
                                                                            orig_keypoints=keypoints,
                                                                            orig_parts=parts,
                                                                            roi_width=roi_width,
                                                                            roi_height=roi_height,
                                                                            dilation_size=dilation_size)

    return new_instances, new_image, new_keypoints, new_parts


def _smooth_edges(orig_img, ksize=5, n_iter=15):
    """Pyramid median filter."""
    blurred_img = cv2.pyrUp(orig_img)
    for _ in range(n_iter):
        blurred_img = cv2.medianBlur(blurred_img, ksize)
    return cv2.pyrDown(blurred_img)


def _smooth_image(image, blur_type='gauss', ksize=9):
    if blur_type == 'gauss':
        blurred_img = cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif blur_type == 'bilateral':
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        blurred_img = cv2.bilateralFilter(image, ksize, 10 * sigma, sigma)
    elif blur_type == 'median':
        blurred_img = cv2.medianBlur(image, 5)
    elif blur_type == 'edge':
        blurred_img = _smooth_edges(image, ksize=5, n_iter=1)
    else:
        raise NotImplementedError
    return blurred_img


def augment_sample(image, instances, bg, keypoints=None, parts=None,
                   warping=True, distortion=15,
                   shift=(0, 0), shift_upper=True,
                   blur_shifted=True, blur_stationary=False, blur_type=None,
                   rot_angle=0, roi_width=(190, 430), roi_height=(10, 410), scale_shifted=1.0,
                   dilation_size=15, seed=0):
    # Check that the parameters are valid
    assert image.shape == bg.shape
    assert image.shape[:2] == instances.shape[:2]
    assert instances.ndim == 3 and instances.shape[-1] == 2
    if parts is not None:
        assert instances.shape == parts.shape
    assert (not warping) or (warping and keypoints is not None)

    # Initialization
    if seed is not None:
        np.random.seed(seed)

    upper_id = 0 if shift_upper else 1
    orig_image = image.copy()

    # Rotate first instance
    if rot_angle != 0:
        instances, image, keypoints, parts = _apply_transform(image, instances, upper_id,
                                                              bg=bg, keypoints=keypoints, parts=parts,
                                                              transform='rotate', rot_angle=rot_angle,
                                                              roi_width=roi_width, roi_height=roi_height,
                                                              dilation_size=dilation_size)

    # Scale first instance
    if scale_shifted != 1.0:
        instances, image, keypoints, parts = _apply_transform(image, instances, upper_id,
                                                              bg=bg, keypoints=keypoints, parts=parts,
                                                              transform='rescale', scale_shifted=scale_shifted,
                                                              roi_width=roi_width, roi_height=roi_height,
                                                              dilation_size=dilation_size)

    # Warp first instance
    if warping:
        instances, image, keypoints, parts = _apply_warping(image, instances, upper_id, keypoints,
                                                            bg=bg, parts=parts, distortion=distortion,
                                                            roi_width=roi_width, roi_height=roi_height,
                                                            dilation_size=dilation_size)

    fst_id, snd_id, fst_kp_channels, snd_kp_channels = _get_channel_ids(upper_id)
    # Shift first instance
    if shift is not None:
        # Calculate move matrix to shift the first rat
        upper_center = ndimage.measurements.center_of_mass(instances[..., fst_id])
        lower_center = ndimage.measurements.center_of_mass(instances[..., snd_id])
        move = (int(lower_center[0] - upper_center[0] + shift[0]),
                int(lower_center[1] - upper_center[1] + shift[1]))
        instances, image, keypoints, parts = _apply_translation(image, instances, upper_id, move,
                                                                bg=bg, keypoints=keypoints, parts=parts,
                                                                roi_width=roi_width, roi_height=roi_height,
                                                                dilation_size=dilation_size)

    # Inpaint lower instance around the upper instance
    dilated_snd_inst = dilation(instances[..., snd_id] > 0, square(5))
    vis_mask = (instances[..., fst_id] == 0) & dilated_snd_inst
    image[vis_mask] = orig_image[vis_mask]

    # Blur shifted instance
    if blur_type is not None and (blur_shifted or blur_stationary):
        # Apply bluring on the whole image
        blurred_img = _smooth_image(image, blur_type=blur_type)

        # Only blur the specified instances, inpaint the rest
        if blur_shifted:
            dilated_instances = dilation(instances[..., fst_id] > 0, square(5))
            image[dilated_instances] = blurred_img[dilated_instances]
        if blur_stationary:
            dilated_instances = dilation(instances[..., snd_id] > 0, square(5))
            image[dilated_instances] = blurred_img[dilated_instances]

    return instances, image, keypoints, parts, fst_id, snd_id


def apply_shadow(image, instances, upper_id, strength=0.25):
    tmp = ((instances == 1) * 255).astype(np.uint8)
    dilated = dilation(tmp, square(3))
    outline = dilated - dilation(tmp, square(1))
    blurred = cv2.GaussianBlur(outline[..., None], (5, 5), 0)
    blurred[instances == 1] = 0

    shadow = np.dstack((blurred, blurred, blurred)) / blurred.max() * strength
    shadow = 1 - shadow
    return np.rint((image * shadow)).astype(np.uint8)


def apply_color_augmentation(images, n_aug_per_img=2):  # number of images to generate (should be <= 10)
    copy_directory(f'{WBAUG_PATH}/params', "./params")

    wb_color_augmenter = wbAug.WBEmulator()
    augmented_images = []
    for img in images:
        aug_imgs, aug_params = wb_color_augmenter.generateWbsRGB(img, n_aug_per_img)
        augmented_images.append([np.array(img) for img in aug_imgs])
    return augmented_images
