import os
import itertools
import multiprocessing

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
    # from sklearn.utils.linear_assignment_ import linear_assignment
from skimage.measure import regionprops, EllipseModel, label
from skimage.morphology import dilation, square
import skimage.draw
import skimage.measure
from skimage.morphology import remove_small_holes, closing, disk
from skimage.morphology import convex_hull_image, remove_small_holes, remove_small_objects
import numpy
from scipy import stats

from phdutils import io, segmentation, superpixel, visualization
import automatic_annotation as aa
import annotation2coco as a2c
import utils.io
import utils.segmentation
import utils.visualization


def mask_intersection_over_union(maskA, maskB):
    """
    maskA: multi-label mask
    maskB: binary mask
    """
    label_ids, intersect_counts = np.unique(maskA[maskB > 0], return_counts=True)
    _, all_counts = np.unique(maskA[np.isin(maskA, label_ids)], return_counts=True)
    ious = intersect_counts / all_counts
    if len(label_ids) == 0:
        return np.asarray([0.]), np.asarray([0])
    elif label_ids[0] == 0:
        if 1 < len(label_ids):
            return ious[1:], label_ids[1:]
        else:
            return np.asarray([0.]), np.asarray([0])
    else:
        return ious, label_ids


def tracktor_results_to_instances(data, shape, allow_overwrite=True):
    annots = []
    instances = np.zeros(shape, dtype=np.uint8)
    for frame_id, results in data.items():
        annots.append({})
        for idx, (inst_id, result) in enumerate(results.items()):
            bbox, confidence, class_id, mask = result[:4]
            x1, y1, x2, y2 = bbox
            tmp = np.zeros_like(instances[int(frame_id)], dtype=np.bool)
            tmp[y1:y2, x1:x2] = np.asarray(mask)

            '''
            old_inst_ids, old_counts = np.unique(instances[int(frame_id), tmp])
            if old_inst_ids[0] == 0:
                old_inst_ids = old_inst_ids[1:]
                old_counts = old_counts[1:]
            '''
            if not allow_overwrite:
                ious, label_ids = mask_intersection_over_union(instances[int(frame_id)], tmp)
                label_mask = (ious > 0.1)
                ious, label_ids = ious[label_mask], label_ids[label_mask]
                tmp[np.isin(instances[int(frame_id)], label_ids)] = False

            instances[int(frame_id), tmp] = (int(inst_id)+1)
            annots[int(frame_id)][str(int(inst_id)+1)] = {'pred_classes': class_id}
    return instances, annots


def tracktor_results_to_keypoints(data):
    annots = []
    for frame_id, results in data.items():
        annots.append({})
        for idx, (inst_id, result) in enumerate(results.items()):
            keypoints = result[-1]
            annots[int(frame_id)][str(int(inst_id)+1)] = {'pred_keypoints': keypoints}
    return annots


def resize_images(imgs, height, width):
    # shape: (W, H)
    shape = list(imgs.shape)
    shape[1], shape[2] = height, width
    resized = np.empty(shape, dtype=np.uint8)
    for idx in range(len(imgs)):
        resized[idx] = cv2.resize(imgs[idx], (width, height))
    return resized


def load_files(seq):
    # Load RGB images
    rgb_dir = '/media/hdd2/lkopi/datasets/rats/tracking_test/test_videos/training_data/images/{}'.format(seq)
    fns = utils.io.list_directory(rgb_dir, sort_key=utils.io.filename_to_number, extension='.png')
    imgs = utils.io.read_arrays(fns, utils.io.read)

    # Load Cluster-RCNN instances
    cluster_fn = '/media/hdd2/lkopi/datasets/rats/tracking_test/tracktor_out/{0}/{0}_cluster_raw.avi'.format(seq)
    clusters = utils.io.read(cluster_fn)

    cinstances = np.zeros(clusters.shape[:-1], dtype=np.uint8)
    cinstances[clusters[..., 1] > 127] = 1
    cinstances[clusters[..., 2] > 127] = 2
    # cinstances = skimage.measure.label(cinstances, connectivity=3)

    # Load keypoints detector instances and annotations
    kp_json = '/media/hdd2/lkopi/datasets/rats/tracking_test/tracktor_out/{0}/{0}_kp_reid.json'.format(seq)
    kp_data = utils.io.read(kp_json)
    instances, _ = tracktor_results_to_instances(kp_data, cinstances.shape, allow_overwrite=False)  # (len(kp_data), 480, 640))
    # instances = skimage.measure.label(instances, connectivity=3)
    # instances = resize_images(instances, height=cinstances.shape[1], width=cinstances.shape[2])
    kp_annots = tracktor_results_to_keypoints(kp_data)

    # Load part detector instances and annotations
    part_json = '/media/hdd2/lkopi/datasets/rats/tracking_test/tracktor_out/{0}/{0}_parts_reid.json'.format(seq)
    part_data = utils.io.read(part_json)
    parts, part_annots = tracktor_results_to_instances(part_data, cinstances.shape, allow_overwrite=False)  # (len(part_data), 480, 640))
    # parts = resize_images(parts, height=cinstances.shape[1], width=cinstances.shape[2])

    return imgs, cinstances, instances, kp_annots, parts, part_annots


def fill_small_holes(labels, area_threshold=50):
    """Fill regions that are smaller than `area_threshold`.

    Parameters
    ----------
    labels : np.ndarray
        2D matrix containing the labels
    area_threshold : int, optional
        Minimum allowed area of each region, by default 50

    Returns
    -------
    np.ndarray
        Label matrix where no regions with smaller than `area_threshold` holes are present.
    """
    labels = labels.copy()
    label_ids = np.unique(labels)[1:]

    for label_id in label_ids:
        mask = (labels == label_id)
        mask = remove_small_holes(mask, area_threshold=area_threshold)
        labels[mask] = label_id
    return labels


def apply_closing(labels, disk_size=6):
    """Fill regions that are smaller than `area_threshold`.

    Parameters
    ----------
    labels : np.ndarray
        2D matrix containing the labels
    area_threshold : int, optional
        Minimum allowed area of each region, by default 50

    Returns
    -------
    np.ndarray
        Label matrix where no regions with smaller than `area_threshold` holes are present.
    """
    labels = labels.copy()
    label_ids = np.unique(labels)[1:]
    selem = disk(disk_size)

    for label_id in label_ids:
        mask = (labels == label_id)
        mask = closing(mask, selem)
        labels[mask] = label_id
    return labels


def get_inst_ids_and_area(frame):
    inst_ids, counts = np.unique(frame, return_counts=True)
    if inst_ids[0] == 0:
        inst_ids, counts = inst_ids[1:], counts[1:]
    return inst_ids, counts


def clean_frame(idx, frame, min_inst_per_frame=None, max_inst_per_frame=None, remove_incosistencies=True,
                remove_small_regions=True, min_area=200, closing_area=10, overlap_thrs=0.1, check_shape=True,
                remove_thorned_regions=False, thorned_thrs=1., multichannel=False, drop_unused_channels=True):
    # Smooth instances
    if remove_small_regions:
        if not multichannel:
            frame = frame[..., None]
        for inst_id in range(frame.shape[-1]):
            frame[..., inst_id] = segmentation.remove_small_regions(frame[..., inst_id], bg_label=0, min_area=min_area)
            if 0 < closing_area and 0 < np.count_nonzero(frame[..., inst_id]):
                frame[..., inst_id] = apply_closing(frame[..., inst_id], disk_size=closing_area)
            frame[..., inst_id] = fill_small_holes(frame[..., inst_id], area_threshold=min_area)
        if not multichannel:
            frame = frame[..., 0]

    inst_ids, counts = get_inst_ids_and_area(frame)
    if min_inst_per_frame is not None and len(inst_ids) < min_inst_per_frame:
        frame *= 0
        return idx, frame

    if check_shape and (min(counts) / max(counts) < 0.4):  # only for 2 instances of the same class
        frame[frame == inst_ids[np.argmin(counts)]] = 0
        # Recalculate instance indices and sizes
        inst_ids, counts = get_inst_ids_and_area(frame)

    if remove_thorned_regions:
        for inst_id in inst_ids:
            inst_mask = (frame == inst_id)
            if multichannel:
                inst_mask = inst_mask[..., inst_id-1]

            if thorned_thrs == 1.:
                _, n_labels = label(inst_mask, connectivity=2, return_num=True)
                if 1 < n_labels:
                    if not multichannel:
                        frame[inst_mask] = 0
                    else:
                        frame[inst_mask, inst_id-1] = 0
            elif 0. < thorned_thrs:
                chull = convex_hull_image(inst_mask)
                if (np.count_nonzero(inst_mask) / np.count_nonzero(chull)) < thorned_thrs:
                    if not multichannel:
                        frame[inst_mask] = 0
                    else:
                        frame[inst_mask, inst_id-1] = 0
        # Recalculate instance indices and sizes
        inst_ids, counts = get_inst_ids_and_area(frame)

    if remove_incosistencies:
        for inst_id in inst_ids:
            inst_mask = (frame == inst_id)
            if 0 == np.count_nonzero(inst_mask):
                continue
            if multichannel:
                inst_mask = inst_mask[..., inst_id-1]
            # inst_mask = remove_small_objects(inst_mask, min_size=256)
            chull = convex_hull_image(inst_mask)
            if multichannel:
                chull = np.stack([chull]*frame.shape[-1], axis=-1)

            tmp = frame.copy()
            if not multichannel:
                tmp[inst_mask] = 0
            else:
                tmp[inst_mask, inst_id-1] = 0

            ious, label_ids = mask_intersection_over_union(tmp, chull)
            label_mask = (ious > overlap_thrs)
            ious, label_ids = ious[label_mask], label_ids[label_mask]
            if len(label_ids) != 0:
                if not multichannel:
                    frame[np.isin(tmp, label_ids)] = 0
                else:
                    for lid in label_ids:
                        frame[..., lid-1] = 0
        # Recalculate instance indices and sizes
        inst_ids, counts = get_inst_ids_and_area(frame)

    if (max_inst_per_frame is not None and len(inst_ids) > max_inst_per_frame):
        frame *= 0

    return idx, frame


def clean_frame_wrapper(params):
    return clean_frame(*params)


def clean_instances_par(instances, min_inst_per_frame=None, max_inst_per_frame=None, remove_incosistencies=True,
                        remove_small_regions=True, min_area=200, closing_area=10, overlap_thrs=0.1, check_shape=True,
                        remove_thorned_regions=False, thorned_thrs=1., multichannel=False):
    instances = instances.copy()
    '''
    for idx, frame in enumerate(instances):
        i, f = clean_frame(idx, frame, min_inst_per_frame, max_inst_per_frame, remove_incosistencies,
                           remove_small_regions, min_area, closing_area, overlap_thrs, check_shape,
                           remove_thorned_regions, thorned_thrs, multichannel)
        instances[i] = f
    '''
    pool = multiprocessing.Pool()
    for idx, frame in pool.imap_unordered(clean_frame_wrapper,
                                          zip(range(len(instances)),
                                              instances,
                                              itertools.repeat(min_inst_per_frame),
                                              itertools.repeat(max_inst_per_frame),
                                              itertools.repeat(remove_incosistencies),
                                              itertools.repeat(remove_small_regions),
                                              itertools.repeat(min_area),
                                              itertools.repeat(closing_area),
                                              itertools.repeat(overlap_thrs),
                                              itertools.repeat(check_shape),
                                              itertools.repeat(remove_thorned_regions),
                                              itertools.repeat(thorned_thrs),
                                              itertools.repeat(multichannel))):
        instances[idx] = frame
    pool.close()
    pool.join()
    return instances


def clean_instances(instances, min_inst_per_frame=None, max_inst_per_frame=None, remove_incosistencies=True,
                    remove_small_regions=True, min_area=200, closing_area=10, overlap_thrs=0.1, check_shape=True,
                    remove_thorned_regions=False, thorned_thrs=1., multichannel=False):
    instances = instances.copy()
    
    for idx, frame in enumerate(instances):
        # Smooth instances
        if remove_small_regions:
            instances[idx] = segmentation.remove_small_regions(instances[idx], bg_label=0, min_area=min_area)
            if 0 < closing_area:
                instances[idx] = apply_closing(instances[idx], disk_size=closing_area)
            instances[idx] = fill_small_holes(instances[idx], area_threshold=min_area)
        # instances[idx] = remove_small_objects(instances[idx], min_size=256)
        # instances[idx] = remove_small_holes(instances[idx], connectivity=4)
        frame = instances[idx]  # just a reference (assert frame == instances[idx])

        inst_ids, counts = get_inst_ids_and_area(frame)
        if min_inst_per_frame is not None and len(inst_ids) < min_inst_per_frame:
            instances[idx] = 0
            continue

        if check_shape and (min(counts) / max(counts) < 0.4):  # only for 2 instances of the same class
            instances[idx,  frame == inst_ids[np.argmin(counts)]] = 0
            # Recalculate instance indices and sizes
            inst_ids, counts = get_inst_ids_and_area(frame)

        if remove_thorned_regions:
            for inst_id in inst_ids:
                inst_mask = (frame == inst_id)

                if thorned_thrs == 1.:
                    _, n_labels = label(inst_mask, connectivity=2, return_num=True)
                    if 1 < n_labels:
                        instances[idx, inst_mask] = 0
                elif 0. < thorned_thrs:
                    chull = convex_hull_image(inst_mask)
                    if (np.count_nonzero(inst_mask) / np.count_nonzero(chull)) < thorned_thrs:
                        instances[idx, inst_mask] = 0
            # Recalculate instance indices and sizes
            inst_ids, counts = get_inst_ids_and_area(frame)

        if remove_incosistencies:
            for inst_id in inst_ids:
                inst_mask = (frame == inst_id)
                # inst_mask = remove_small_objects(inst_mask, min_size=256)
                chull = convex_hull_image(inst_mask)

                tmp = frame.copy()
                tmp[inst_mask] = 0

                ious, label_ids = mask_intersection_over_union(tmp, chull)
                label_mask = (ious > overlap_thrs)
                ious, label_ids = ious[label_mask], label_ids[label_mask]
                if len(label_ids) != 0:
                    instances[idx, np.isin(tmp, label_ids)] = 0
                    break
            # Recalculate instance indices and sizes
            inst_ids, counts = get_inst_ids_and_area(frame)

        if (max_inst_per_frame is not None and len(inst_ids) > max_inst_per_frame):
            instances[idx] = 0
        assert (frame == instances[idx]).all()
    return instances


def get_data(seq):
    imgs, cinstances, instances, kp_annots, parts, part_annots = load_files(seq)

    instances = clean_instances(instances, min_inst_per_frame=2)
    cinstances = clean_instances(cinstances, min_inst_per_frame=2)
    parts = clean_instances(parts, min_area=100, closing_area=6)
    return imgs, cinstances, instances, parts, kp_annots, part_annots


def plot_keypoints(imgs, kp_annots, colors=((255,0,0),(0,255,0),(0,0,255)), use_conf=True): # head, body, tail
    imgs = imgs.copy()
    for frame_id, kp_anns in enumerate(kp_annots):
        for inst_id, anns in kp_anns.items():
            for i, kp in enumerate(anns['pred_keypoints']):
                x, y, conf = kp
                if not use_conf or 0 < conf:
                    imgs[frame_id] = cv2.circle(imgs[frame_id], center=(x, y), radius=2, color=colors[i], thickness=2)
    return imgs


def improve_keypoints(kp_annots, dist_threshold=50, instances=None, parts=None, part_annots=None):
    prev_anns = kp_annots[0]
    for frame_id, kp_anns in enumerate(kp_annots):
        for inst_id, anns in kp_anns.items():
            if 0 == frame_id:
                # Set visibility
                kps = np.asarray(anns['pred_keypoints'])
                kps[:, 2] = 1
                anns['pred_keypoints'] = kps.tolist()
            else:
                if instances is not None and int(inst_id) not in instances[frame_id]:
                    kps = np.asarray(anns['pred_keypoints'])
                    kps[:, 2] = 0
                    anns['pred_keypoints'] = kps.tolist()
                    break

                if inst_id not in prev_anns:
                    # Reset new keypoints
                    kps = np.asarray(anns['pred_keypoints'])
                    kps[:, 2] = 1
                    anns['pred_keypoints'] = kps.tolist()
                else:
                    prev_vis = np.asarray(prev_anns[inst_id]['pred_keypoints'])[:, 2]

                    prev_kps = np.asarray(prev_anns[inst_id]['pred_keypoints'])[:, :2]
                    curr_kps = np.asarray(anns['pred_keypoints'])[:, :2]
                    dist = (prev_kps - curr_kps)**2
                    dist = np.sqrt(dist[:, 0] + dist[:, 1])
                    dist_mask = (dist < dist_threshold) & prev_vis

                    rev_dist = (prev_kps - curr_kps[::-1, :])**2
                    rev_dist = np.sqrt(rev_dist[:, 0] + rev_dist[:, 1])
                    rev_dist_mask = (rev_dist < dist_threshold) & prev_vis
                    # print(dist, rev_dist, prev_kps, curr_kps)

                    kps = np.asarray(anns['pred_keypoints'])

                    reverse_kps = (np.sum(rev_dist_mask) > np.sum(dist_mask))
                    if reverse_kps:
                        # print(frame_id, rev_dist_mask, dist_mask)
                        kps = kps[::-1, :]
                        dist_mask = rev_dist_mask

                    if 0 < np.sum(dist_mask):
                        # Use old keypoints, shifted in the same direction if we lost it
                        kps[:, 2] = dist_mask*1

                        shift = kps[dist_mask, :2] - prev_kps[dist_mask]
                        shift = np.average(shift, axis=0).astype(np.int32)
                        prev_kps += shift
                        prev_kps[prev_kps < 0] = 0
                        prev_kps[640 <= prev_kps[:, 0], 0] = 639
                        prev_kps[420 <= prev_kps[:, 1], 1] = 419

                        kps[np.logical_not(dist_mask), :2] = prev_kps[np.logical_not(dist_mask)]
                        anns['pred_keypoints'] = kps.tolist()
                    else:
                        # Reset new keypoints
                        kps = np.asarray(anns['pred_keypoints'])
                        kps[:, 2] = 1
                        anns['pred_keypoints'] = kps.tolist()
                if parts is not None and part_annots is not None:
                    for i, kp in enumerate(anns['pred_keypoints']):
                        x, y, conf = kp
                        # if 0 < conf:
                        # part_inst_id = parts[frame_id, y, x]
                        part_inst_ids = parts[frame_id, max(0, y-15):y+15, max(0, x-15):x+15]
                        part_inst_ids, counts = np.unique(part_inst_ids, return_counts=True)
                        if 0 < len(part_inst_ids) and part_inst_ids[0] == 0:
                            part_inst_ids, counts = part_inst_ids[1:], counts[1:]
                        if 0 < len(part_inst_ids):
                            part_inst_id = str(part_inst_ids[np.argmax(counts)])
                            # print(part_inst_id)
                            anns['pred_keypoints'][i][2] = (part_inst_id in part_annots[frame_id] and (part_annots[frame_id][part_inst_id]['pred_classes'] == i))
                            # print(anns['pred_keypoints'][i][2], part_inst_id, part_inst_id in part_annots[frame_id], part_inst_id in part_annots[frame_id] and part_annots[frame_id][part_inst_id]['pred_classes'])
                        else:
                            anns['pred_keypoints'][i][2] = 0
        prev_anns = kp_anns
    return kp_annots


def parts2instances(parts, guide_insts):
    instances = np.zeros_like(guide_insts)
    inst_ids = np.unique(guide_insts)
    for inst_id in inst_ids:
        tmp = utils.segmentation.foreground_proposal(parts, guide_insts == inst_id)[0]
        instances[tmp > 0] = tmp[tmp > 0] * inst_id
    return instances


def refine_instances(instance_mask):
    assert instance_mask.ndim == 2
    selem = disk(10)
    instance_mask = closing(remove_small_holes(instance_mask > 0, connectivity=4), selem)
    encoded_masks = utils.segmentation.encode_mask(instance_mask)
    if len(encoded_masks) == 0:
        return instance_mask
    instance_mask = utils.segmentation.decode_mask(encoded_masks[0], instance_mask.shape)
    return instance_mask


def improve_masks(masks):
    outs = np.zeros_like(masks)
    inst_ids = np.unique(masks)

    for i, mask in enumerate(masks):
        for inst_id in inst_ids:
            if inst_id in mask:
                tmp = (mask == inst_id)
                tmp = refine_instances(tmp)
                outs[i, tmp] = inst_id
    return outs



# source: https://stackoverflow.com/a/35674754
def mode(ndarray, axis=0):
    # Check inputs
    ndarray = numpy.asarray(ndarray)
    ndim = ndarray.ndim
    if ndarray.size == 1:
        return (ndarray[0], 1)
    elif ndarray.size == 0:
        raise Exception('Cannot compute mode on empty array')
    try:
        axis = range(ndarray.ndim)[axis]
    except:
        raise Exception('Axis "{}" incompatible with the {}-dimension array'.format(axis, ndim))

    # If array is 1-D and numpy version is > 1.9 numpy.unique will suffice
    if all([ndim == 1,
            int(numpy.__version__.split('.')[0]) >= 1,
            int(numpy.__version__.split('.')[1]) >= 9]):
        modals, counts = numpy.unique(ndarray, return_counts=True)
        index = numpy.argmax(counts)
        return modals[index], counts[index]

    # Sort array
    sort = numpy.sort(ndarray, axis=axis)
    # Create array to transpose along the axis and get padding shape
    transpose = numpy.roll(numpy.arange(ndim)[::-1], axis)
    shape = list(sort.shape)
    shape[axis] = 1
    # Create a boolean array along strides of unique values
    strides = numpy.concatenate([numpy.zeros(shape=shape, dtype='bool'),
                                 numpy.diff(sort, axis=axis) == 0,
                                 numpy.zeros(shape=shape, dtype='bool')],
                                axis=axis).transpose(transpose).ravel()
    # Count the stride lengths
    counts = numpy.cumsum(strides)
    counts[~strides] = numpy.concatenate([[0], numpy.diff(counts[~strides])])
    counts[strides] = 0
    # Get shape of padded counts and slice to return to the original shape
    shape = numpy.array(sort.shape)
    shape[axis] += 1
    shape = shape[transpose]
    slices = [slice(None)] * ndim
    slices[axis] = slice(1, None)
    # Reshape and compute final counts
    counts = counts.reshape(shape).transpose(transpose)[slices] + 1

    # Find maximum counts and return modals/counts
    slices = [slice(None, i) for i in sort.shape]
    del slices[axis]
    index = numpy.ogrid[slices]
    index.insert(axis, numpy.argmax(counts, axis=axis))
    return sort[index], counts[index]

def merge_tracks(list_of_instances):
    n_instances = len(list_of_instances)
    merged = np.asarray(list_of_instances).reshape(n_instances, -1)
    # merged = stats.mode(merged, axis=0)[0]
    merged = mode(merged, axis=0)[0]
    return merged.reshape(list_of_instances[0].shape)

def relabel_instances(curr_instances, guide_insts):
    # relabel track, if one has disappeared for the rest of the video?
    outs = curr_instances.copy()
    for i, frame in enumerate(curr_instances):
        inst_ids = np.unique(guide_insts[i])
        for inst_id in inst_ids:
            ious, label_ids = mask_intersection_over_union(frame, guide_insts[i] == inst_id)
            iou_mask = (ious > 0.1)
            if 0 < np.sum(iou_mask):
                label_id = label_ids[np.argmax(ious)]
                outs[i, frame==label_id] = inst_id
            # tmp = segmentation.foreground_proposal(frame, guide_insts[i] == inst_id)[0]
            # outs[i, tmp > 0] = tmp[tmp > 0] * inst_id
    return outs


def run(seq):
    imgs, cinstances, instances, parts, kp_annots, part_annots = get_data(seq)

    rgb_cinstances = utils.visualization.map_labels(cinstances, color_map, dtype=np.uint8)
    rgb_instances = utils.visualization.map_labels(instances, color_map, dtype=np.uint8)
    rgb_parts = utils.visualization.map_labels(parts, color_map, dtype=np.uint8)

    kp_annots = improve_keypoints(kp_annots, instances=instances, parts=parts, part_annots=part_annots)
    imgs_ = plot_keypoints(imgs, kp_annots, use_conf=True)


    part_instances = improve_masks(parts2instances(parts, instances))
    part_cinstances = improve_masks(parts2instances(parts, cinstances))

    rgb_part_instances = utils.visualization.map_labels(part_instances, color_map, dtype=np.uint8)
    rgb_part_cinstances = utils.visualization.map_labels(part_cinstances, color_map, dtype=np.uint8)


    instances_ = relabel_instances(instances, part_instances)
    cinstances_ = relabel_instances(cinstances, part_cinstances)

    rgb_instances_ = utils.visualization.map_labels(instances_, color_map, dtype=np.uint8)
    rgb_cinstances_ = utils.visualization.map_labels(cinstances_, color_map, dtype=np.uint8)

    only_insts = merge_tracks([instances_, cinstances_])
    only_parts = merge_tracks([part_instances, part_cinstances])
    all_instances = merge_tracks([part_instances, part_cinstances, instances_, cinstances_])

    only_insts = utils.visualization.map_labels(only_insts, color_map, dtype=np.uint8)
    only_parts = utils.visualization.map_labels(only_parts, color_map, dtype=np.uint8)
    all_instances = utils.visualization.map_labels(all_instances, color_map, dtype=np.uint8)


    bitwise_merge = (part_instances | part_cinstances | instances_ | cinstances_)
    bitwise_merge = utils.visualization.map_labels(bitwise_merge, color_map, dtype=np.uint8)

    out = utils.visualization.concat_images([imgs_, rgb_cinstances_, rgb_instances_,
                                             rgb_parts, rgb_part_instances, rgb_part_cinstances,
                                             bitwise_merge, only_parts, all_instances], n_rows=3)

    return out, kp_annots, part_annots, seq

'''
np.random.seed(0)
colors = visualization.generate_random_colors(100)
color_map = dict(zip(range(len(colors)), colors))
color_map[0] = [255,255,255]

out_dir = io.datestring() + '_baseline'
base_dir = '/media/hdd2/lkopi/datasets/rats/tracking_test/test_videos/training_data/images'
seqs = io.list_directory(base_dir, only_dirs=True, full_path=False)
# seq = 'hard_0_fin'
pool = multiprocessing.Pool()
for out, kp_annots, part_annots, seq in pool.imap_unordered(run, seqs):
    io.save_video(out_dir + '/' + seq + '.avi', out)
    io.save(out_dir + '/' + seq + '_kp.json', kp_annots)
    io.save(out_dir + '/' + seq + '_bp.json', part_annots)

# for seq in seqs:
#     out, kp_annots, part_annots = run(seq)
'''