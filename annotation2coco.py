#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import itertools
import math
import multiprocessing
import os
import random
import shutil
import sys

from detectron2.structures import BoxMode
import numpy as np
from pycocotools import mask as cocomask
from tqdm import tqdm

WBAUG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          'external/WB_color_augmenter/WBAugmenter_Python/')
sys.path.append(WBAUG_PATH)
from WBAugmenter import WBEmulator as wbAug

import augmentation
import utils.io
import utils.keypoints
import utils.segmentation


def circle_sections(divisions, radius=1):
    """Get N equally spaced translations around a circle with the specified radius.

    Parameters
    ----------
    divisions : int
        Number of equally spaced translation to get.
    radius : int, optional
        Radious of the circle, by default 1

    Returns
    -------
    list of int
        (N,2) list of coordinates.
    """
    # the difference between angles in radians -- don't bother with degrees
    angle = 2 * math.pi / divisions
    # a list of all angles using a list comprehension
    angles = [i*angle for i in range(divisions)]
    # finally return the coordinates on the circle as a list of 2-tuples
    return [(radius*math.cos(a), radius*math.sin(a)) for a in angles]


def determine_bbox(mask):
    """Determine the bounding box of the given mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.

    Returns
    -------
    ((float, float, float, float), float)
        Coordinates of the bounding box (x, y, w, h) and the area of the segmentation.
    """
    fortran_binary_mask = np.asfortranarray(mask.astype(np.uint8))
    encoded_mask = cocomask.encode(fortran_binary_mask)

    area = cocomask.area(encoded_mask)
    bounding_box = cocomask.toBbox(encoded_mask)
    return bounding_box, area


def coco_segmentation(gt_mask):
    bounding_box, area = determine_bbox(gt_mask)
    segmentations = utils.segmentation.encode_mask(gt_mask)
    return segmentations, area.tolist(), bounding_box.tolist()


def fit_bbox_on_keypoint(gt_mask, keypoint, bbox_size):
    masked_part = np.zeros_like(gt_mask)
    h, w = masked_part.shape
    masked_part[max(0, keypoint[0] - bbox_size):min(h, keypoint[0] + bbox_size),
                max(0, keypoint[1] - bbox_size):min(w, keypoint[1] + bbox_size)] = \
        gt_mask[max(0, keypoint[0] - bbox_size):min(h, keypoint[0] + bbox_size),
                max(0, keypoint[1] - bbox_size):min(w, keypoint[1] + bbox_size)]
    return coco_segmentation(masked_part)


def coco_segmentation_part(gt_mask, keypoints, bbox_size=40): # appr. 35x35 (or 40x40) bboxes would be enough => fit to size
    ground_truth_binary_mask = gt_mask.astype(np.uint8)

    part_segments, part_areas, part_bboxes = {}, {}, {}
    for key, kp in keypoints.items():
        part_segments[key], part_areas[key], part_bboxes[key] = fit_bbox_on_keypoint(ground_truth_binary_mask, kp[-2:], bbox_size)

    return part_segments, part_areas, part_bboxes


def anns2kps(anns, shape):
    h, w = shape[:2]
    htmps = [None] * max([int(k) for k in anns.keys()])
    for inst_id in range(len(htmps)):
        if str(inst_id) in anns:
            htmps[inst_id] = utils.keypoints.keypoints2heatmap(list(anns[str(inst_id)].values()), shape)
        else:
            htmps[inst_id] = np.zeros((h, w, 3), dtype=np.uint8)
    return np.dstack(htmps)


def kps2anns(kp_htmps, instances=None, part_classes=None):
    assert kp_htmps.shape[-1] % 3 == 0

    kp_dict = {}
    for inst_id in range(kp_htmps.shape[-1] // 3):
        kps = utils.keypoints.heatmap2keypoints(kp_htmps[..., inst_id * 3:(inst_id + 1) * 3])
        missing_keypoints = len(list(filter(lambda l: l == [], kps))) > 0
        if not missing_keypoints:
            if instances is not None:
                for kp in kps:
                    c, r = kp
                    kp.insert(0, 1+int(instances[c, r] == inst_id))
            if part_classes is not None:
                for class_id, kp in enumerate(kps):
                    c, r = kp[-2:]
                    if class_id != 1:  # TODO: not too nice fix for tail_base
                        visibility = 1+int(part_classes[c, r] == class_id+1)
                    else:  # if it's tail base, then it is either body or tail
                        visibility = 1+int(part_classes[c, r] >= class_id+1)
                    if len(kp) == 2:
                        kp.insert(0, visibility)
                    else:
                        kp[0] *= visibility-1
            kps = dict(zip(('head', 'tail_base', 'tail'), kps))
            kp_dict[str(inst_id)] = kps
    return kp_dict


def prep_data(instances, parts=None, anns=None):
    tmp = instances
    inst_ids = np.unique(tmp)[1:]
    instances = np.zeros((tmp.shape[0], tmp.shape[1], 2), dtype=np.bool)
    instances[..., 0] = (tmp == inst_ids[0])
    instances[..., 1] = (tmp >= inst_ids[1])

    if parts is not None:
        tmp = parts
        parts = np.zeros((tmp.shape[0], tmp.shape[1], 2), dtype=np.uint8)
        parts[instances[..., 0], 0] = tmp[instances[..., 0]]
        parts[instances[..., 1], 1] = tmp[instances[..., 1]]

    kp_htmps = None
    if anns is not None:
        kp_htmps = anns2kps(anns, tmp.shape)

    return instances, parts, kp_htmps


def prep_output(instances, parts=None, kp_htmps=None, upper_id=0, lower_id=1):
    tmp = instances
    instances = np.zeros((tmp.shape[0], tmp.shape[1]), dtype=np.uint8)
    instances[tmp[..., lower_id]] = lower_id+1
    instances[tmp[..., upper_id]] = upper_id+1

    if parts is not None:
        tmp = parts
        parts = np.zeros((tmp.shape[0], tmp.shape[1]), dtype=np.uint8)
        parts[instances == lower_id+1] = tmp[instances == lower_id+1, lower_id]
        parts[instances == upper_id+1] = tmp[instances == upper_id+1, upper_id]

    keypoints = kps2anns(kp_htmps, instances=instances) if kp_htmps is not None else None

    return instances, parts, keypoints


def augment_frame(image, instances, bg, shift, aug_params=None, parts=None, anns=None, seed=None):
    if aug_params is None:
        aug_params = augmentation.AugmentParams()

    instances, parts, keypoint_htmps = prep_data(instances, parts, anns)

    if seed is not None:
        np.random.seed(seed)
        aug_seed = np.random.randint(999)
    else:
        aug_seed = None

    new_instances, new_image, new_keypoint_htmps, new_parts, upper_id, lower_id = \
        augmentation.augment_sample(image, instances, bg, keypoints=keypoint_htmps, parts=parts,
                                    shift=shift, shift_upper=np.random.rand() < 0.5,
                                    warping=aug_params.warping, distortion=aug_params.distortion,
                                    blur_type=aug_params.blur_type, blur_stationary=aug_params.blur_stationary,
                                    blur_shifted=aug_params.blur_shifted,
                                    rot_angle=np.random.uniform(-aug_params.rot_angle, aug_params.rot_angle),
                                    roi_width=aug_params.roi_width, roi_height=aug_params.roi_height,
                                    scale_shifted=1 + (np.random.rand() * (aug_params.scale_shifted - 1)),
                                    dilation_size=aug_params.dilation_size, seed=aug_seed)

    new_instances, new_parts, new_keypoints = prep_output(new_instances, new_parts, new_keypoint_htmps, upper_id, lower_id)
    if aug_params.apply_shadow:
        new_image = augmentation.apply_shadow(new_image, new_instances, upper_id)

    return new_image, new_instances, new_parts, new_keypoints


# Based on 2020_06_23_coco_annotation_for_centertrack_IMP.ipynb
class COCOAnnotator():
    def __init__(self, height=420, width=640, start_id=0, first_category_id=0):
        self.height, self.width = height, width
        self.first_category_id = first_category_id
        self._img_id = start_id - 1
        self.part2category = {'head': first_category_id + 0,
                              'body': first_category_id + 1,
                              'tail': first_category_id + 2}
        self.part2pid = {'head': 3, 'body': 2, 'tail': 1}

        self.keypoint_records = []
        self.part_records = []

        utils.io.copy_directory(f'{WBAUG_PATH}/params', "./params")
        self.color_augmenter = wbAug.WBEmulator()
        self.n_aug_per_img = 2  # number of images to generate (should be <= 10)

    def __len__(self):
        return len(self.keypoint_records)

    def _add_image(self, fn):
        assert os.path.isfile(fn) and os.path.exists(fn), fn
        self._img_id += 1
        img = utils.io.read(fn)

        record = {
            "file_name": fn,
            "height": img.shape[0],
            "width": img.shape[1],
            "image_id": self._img_id,
            "annotations": []
        }
        self.keypoint_records.append(record)
        self.part_records.append(copy.deepcopy(record))

    def _add_instance(self, instance_mask, keypoints=None):
        assert keypoints is None or ('head' in keypoints and 'tail_base' in keypoints and 'tail' in keypoints)
        segmentations, area, bbox = coco_segmentation(instance_mask)
        if keypoints is not None:
            tail_kp = keypoints['tail' if 'tail_end' not in keypoints else 'tail_end'][::-1]
            if len(keypoints['head']) == 2:
                kp_list = list(map(float, keypoints['head'][::-1] + [2.0] + keypoints['tail_base'][::-1] + [2.0] + tail_kp + [2.0]))
            else:
                kp_list = list(map(float, keypoints['head'][::-1] + keypoints['tail_base'][::-1] + tail_kp))
        else:
            kp_list = []
        segmentations = list(filter(lambda l: 6 <= len(l), segmentations))
        if area == 0 or not isinstance(segmentations[0], list) or (keypoints is not None and len(kp_list) != 9):
            return

        annot = {
            "bbox": bbox,  # XYWH
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": segmentations,
            "area": area,  # Not needed
            "category_id": self.first_category_id,
            "keypoints": kp_list,
            "iscrowd": 0
        }
        if len(segmentations) != 0:
            self.keypoint_records[-1]["annotations"].append(annot)

    def _add_part(self, part_mask, part_name):
        segmentations, area, bbox = coco_segmentation(part_mask)
        if area == 0:
            return

        annot = {
            "bbox": bbox,  # XYWH
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": segmentations,
            "area": area,  # Not needed
            "category_id": self.part2category[part_name],
            "iscrowd": 0
        }
        if len(segmentations) != 0:
            self.part_records[-1]["annotations"].append(annot)

    def _apply_color_augmentation(self, img_fn):
        base_fn, ext = os.path.splitext(img_fn)
        aug_imgs, aug_params = self.color_augmenter.generateWbsRGB(utils.io.read(img_fn), self.n_aug_per_img)
        aug_fns = []
        for img, param in zip(aug_imgs, aug_params):
            aug_fn = f"{base_fn}{param}{ext}"
            utils.io.save(aug_fn, img)
            aug_fns.append(aug_fn)
        return aug_fns

    def add_sample(self, img_fn, instances, parts=None, anns=None, apply_color_augmentation=False):
        assert parts is None or instances.shape == parts.shape

        if instances.max() >= 254:
            instances //= 127

        img_fns = [img_fn]
        if apply_color_augmentation:
            img_fns += self._apply_color_augmentation(img_fn)

        for img_fn in img_fns:
            self._add_image(img_fn)

            if parts is not None and anns is not None:
                for inst_id, keypoints in anns.items():
                    instance_mask = (instances == int(inst_id)+1).astype(np.uint8)
                    self._add_instance(instance_mask, keypoints)

                    for part_name in self.part2category.keys():
                        part_id = self.part2pid[part_name]
                        part_mask = (instance_mask * (parts == part_id)).astype(np.uint8)
                        self._add_part(part_mask, part_name)

                if 2 != len(self.keypoint_records[-1]["annotations"]) or 6 != len(self.part_records[-1]["annotations"]):
                    self.keypoint_records = self.keypoint_records[:-1]
                    self.part_records = self.part_records[:-1]
                    self._img_id -= 1
            else:
                for inst_id in np.unique(instances)[1:]:
                    instance_mask = (instances == inst_id).astype(np.uint8)
                    self._add_instance(instance_mask)

    def save_annotations(self, kp_json_fn, part_json_fn):
        for fn, data in [(kp_json_fn, self.keypoint_records), (part_json_fn, self.part_records)]:
            utils.io.save(fn, data)

    def save_images(self, out_dir):
        utils.io.make_directory(out_dir)

        for record in self.keypoint_records:
            seq_dir = os.path.basename(os.path.dirname(record["file_name"]))
            base_fn = os.path.basename(record["file_name"])
            orig_fn = os.path.join(record["file_name"])
            new_fn = os.path.join(out_dir, seq_dir, base_fn)
            shutil.copyfile(orig_fn, new_fn)

    @staticmethod
    def merge(a1, a2):
        # Check parameters
        assert a1.height == a2.height
        assert a1.width == a2.width
        assert a1.first_category_id == a2.first_category_id

        # Merge instances
        annotator = COCOAnnotator(height=a1.height, width=a2.width,
                                  start_id=len(a1) + len(a2),
                                  first_category_id=a1.first_category_id)
        annotator.keypoint_records = a1.keypoint_records + a2.keypoint_records
        annotator.part_records = a1.part_records + a2.part_records

        return annotator


def _generate_sections(include_inplace=False, n_sections=5, max_distance=42):
    aug_sections = []
    if include_inplace:
        aug_sections.append(None)
    for rad, n_sample in zip(range(1, max_distance, (max_distance-2)//(n_sections-1)),
                             [1]*n_sections):
        sections = circle_sections(rad, rad)
        for section in random.sample(sections, n_sample):
            aug_sections.append(section)
    return aug_sections


def augment_sample(idx, anns, img_fn, inst_fn, part_fn,
                   aug_params=None, bg=None, shift_parameters=None,
                   aug_img_dir=None, aug_mask_dir=None, seed=None):
    if 2 > len(anns):
        return None

    # Load instances and body-parts
    instances = utils.io.read(inst_fn, -1)
    if np.any(instances >= 127):
        instances //= 127
    parts = utils.io.read(part_fn, -1)

    # Append original sample
    out = [(img_fn, instances, parts, anns)]

    if aug_params is not None:
        img = utils.io.read(img_fn)
        for aug_id, shift in enumerate(shift_parameters):
            # Augment sample
            try:
                aug_image, aug_instances, aug_parts, aug_anns = augment_frame(img, instances, bg, shift, aug_params,
                                                                              parts=parts, anns=anns, seed=seed)
            except Exception as e:
                print('[ERROR]: {} {}'.format(img_fn, e))
                continue

            # Save outputs
            aug_img_fn = os.path.join('{}_{}'.format(aug_img_dir, aug_id), '{}.png'.format(idx))
            utils.io.save(aug_img_fn, aug_image)

            aug_inst_fn = os.path.join('{}_{}'.format(aug_mask_dir, aug_id), '{}_instances.png'.format(idx))
            utils.io.save(aug_inst_fn, aug_instances)

            aug_part_fn = os.path.join('{}_{}'.format(aug_mask_dir, aug_id), '{}_parts.png'.format(idx))
            utils.io.save(aug_part_fn, aug_parts)

            # Append augmented sample
            out.append((aug_img_fn, aug_instances, aug_parts, aug_anns))
    return out


def augment_sample_wrapper(params):
    idx = params[0]
    return augment_sample(*params), idx


def augment_sequence(seq, aug_params, exp_name, img_dir_base, bg_dir_base, annot_dir_base,
                     mask_dir_base=None, temporal=False, overwrite=False, max_bg_id=None):
    # TODO COCOAnnotator does not support temporal augmentation yet...
    print('Sequence id: {}'.format(seq.upper()))
    img_anns = utils.io.read(os.path.join(annot_dir_base, seq).rstrip('/') + '.json')
    if max_bg_id is None:
        max_bg_id = len(os.listdir(os.path.join(bg_dir_base))) - 1

    img_dir, annot_dir = os.path.join(img_dir_base, seq), os.path.join(annot_dir_base, seq)
    mask_dir = os.path.join(mask_dir_base, seq) if mask_dir_base is not None else None

    get_shift_parameters = lambda: None
    aug_img_dir, aug_mask_dir, seed, bg = None, None, None, None
    if aug_params is not None:
        # Select a proper background for inpainting
        seq_int = int(seq) if seq != '' else 0
        bg_id = 230 * (seq_int // 10) + np.random.randint(100) - 50
        if bg_id < 0: bg_id = 0
        if max_bg_id < bg_id: bg_id = max_bg_id
        bg_fn = os.path.join(bg_dir_base, 'bg_' + str(bg_id) + '.png')
        bg = utils.io.read(bg_fn)

        # Set seed and generate sections
        if temporal:
            seed = np.random.randint(9999)
            shift_parameters = _generate_sections(include_inplace=True, n_sections=3)
            get_shift_parameters = lambda: shift_parameters
        else:
            get_shift_parameters = lambda: _generate_sections()
        n_shift_parameters = len(get_shift_parameters())

        # Create directories for the augmented results
        aug_img_dir = os.path.join(img_dir_base, '{}_aug_{}'.format(exp_name, seq))
        aug_mask_dir = os.path.join(annot_dir_base, '{}_aug_{}'.format(exp_name, seq))
        for aug_id in range(n_shift_parameters):
            utils.io.make_directory(os.path.join('{}_{}'.format(aug_img_dir, aug_id)))
            utils.io.make_directory(os.path.join('{}_{}'.format(aug_mask_dir, aug_id)))

        # Initialize keypoint annotation list
        aug_img_anns_list = []
        for _ in range(n_shift_parameters):
            aug_img_anns_list.append({})

    annotator = COCOAnnotator()
    indices, kp_annots = zip(*list(img_anns.items()))
    shift_parameters = [get_shift_parameters() for i in range(len(indices))]
    with multiprocessing.Pool() as pool:
        with tqdm(total=len(indices)) as pbar:
            '''for idx, anns in img_anns.items():
                aug_annots = augment_sample(idx, anns, img_dir, annot_dir, mask_dir=mask_dir,
                                            aug_params=aug_params, bg=bg, shift_parameters=get_shift_parameters(),
                                            aug_img_dir=aug_img_dir, aug_mask_dir=aug_mask_dir, seed=seed)'''

            img_fns = utils.io.list_directory(img_dir, sort_key=utils.io.filename_to_number)
            if mask_dir is not None:
                inst_fns = utils.io.list_directory(mask_dir, sort_key=utils.io.filename_to_number)
            else:
                inst_fns = utils.io.list_directory(annot_dir, sort_key=lambda fn: int(fn.split('_')[0]),
                                                   fn_constraint=lambda fn: fn.endswith('_instances.png'))
            part_fn = utils.io.list_directory(annot_dir, sort_key=lambda fn: int(fn.split('_')[0]),
                                              fn_constraint=lambda fn: fn.endswith('_parts.png'))

            for aug_annots, idx in pool.imap_unordered(augment_sample_wrapper,
                                                       zip(indices, kp_annots, img_fns, inst_fns, part_fn,
                                                           itertools.repeat(aug_params), itertools.repeat(bg),
                                                           shift_parameters, itertools.repeat(aug_img_dir),
                                                           itertools.repeat(aug_mask_dir), itertools.repeat(seed))):
                if temporal and (aug_annots is None or len(aug_annots) <= n_shift_parameters):
                    # Ignore sequence if cannot annotate the every frame
                    for d in [aug_img_dir, aug_mask_dir]:
                        for aug_id in range(n_shift_parameters):
                            os.system('rm -rfd {}_{}'.format(d, aug_id))
                    return COCOAnnotator()
                elif aug_annots is not None:
                    for aug_id, (img_fn, instances, parts, anns) in enumerate(aug_annots):
                        annotator.add_sample(img_fn, instances, parts, anns)    # TODO: this is the bottleneck
                        if 0 < aug_id:
                            aug_img_anns_list[aug_id - 1][idx] = anns
                pbar.update()

    # Save keypoint annotations
    if aug_params is not None:
        for aug_id in range(n_shift_parameters):
            utils.io.save('{}_{}.json'.format(aug_mask_dir, aug_id), aug_img_anns_list[aug_id])

    return annotator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', required=True, help='Experiment name, uses this as a postfix for the generated augmentations and identifies the annotation file.')
    parser.add_argument('-i', '--img_dir_base', required=True, help='Base image directory.')
    parser.add_argument('-b', '--bg_dir_base', required=True, help='Base background directory.')
    parser.add_argument('-a', '--annot_dir_base', required=True, help='Base annotation directory.')
    parser.add_argument('-o', '--out_json_dir', required=True, help='Directory to save the generate COCO json annotation files.')
    parser.add_argument('-m', '--mask_dir_base', help='Base mask directory.')
    parser.add_argument('-p', '--augment_params', help='Augmentation parameters given by a json.')
    parser.add_argument('--temporal', default=False, action='store_true', help='Apply the same augmentation for all samples in a sequence.')
    parser.add_argument('--overwrite', default=False, action='store_true', help='Whether to overwrite the outputs or not.')
    parser.add_argument('--no_seq_folder', default=False, action='store_true',
                        help='Whether the mask directory contains seq folders or not.')
    return parser.parse_args()


def run(exp_name, img_dir_base, bg_dir_base, annot_dir_base, out_json_dir,
        mask_dir_base=None, augment_params=None, temporal=False, overwrite=False, no_seq_folder=False):
    # Validate parameters
    assert os.path.isdir(img_dir_base)
    assert os.path.isdir(bg_dir_base)
    assert os.path.isdir(annot_dir_base)
    assert mask_dir_base is None or os.path.isdir(mask_dir_base)
    assert augment_params is None or (os.path.isfile(augment_params) and os.path.splitext(augment_params)[-1] == '.json')

    # Initialization
    utils.io.make_directory(out_json_dir)
    aug_params = None
    max_bg_id = None
    if augment_params is not None:
        np.random.seed(0)
        aug_params = augmentation.AugmentParams(**utils.io.read(augment_params))
        max_bg_id = len(os.listdir(os.path.join(bg_dir_base)))

    # Process sequences
    annotator = COCOAnnotator()
    seqs = [''] if no_seq_folder else [dn for dn in os.listdir(os.path.join(img_dir_base)) if dn.find('aug_') == -1]
    for seq in seqs:
        annotator = COCOAnnotator.merge(annotator, augment_sequence(seq, aug_params, exp_name, img_dir_base, bg_dir_base,
                                                                    annot_dir_base, mask_dir_base=mask_dir_base,
                                                                    temporal=temporal, overwrite=overwrite,
                                                                    max_bg_id=max_bg_id))

    base_fn = os.path.join(out_json_dir, exp_name)
    keypoint_json_file = base_fn + '_keypoint.json'
    part_json_file = base_fn + '_part.json'
    annotator.save_annotations(keypoint_json_file, part_json_file)


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))