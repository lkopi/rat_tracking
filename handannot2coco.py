import argparse
import os

from detectron2.structures import BoxMode
import numpy as np
from skimage.measure import regionprops

import annotation2coco as a2c
import utils.io
import utils.segmentation
import utils.visualization


def dist(p1, p2):
    # source: https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    a, b = np.array(p1), np.array(p2)
    return np.linalg.norm(1-b)


def rect_distance(bbox1, bbox2):
    # source: https://stackoverflow.com/questions/4978323/how-to-calculate-distance-between-two-rectangles-context-a-game-in-lua
    (x1, y1, x1b, y1b), (x2, y2, x2b, y2b) = bbox1, bbox2
    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0.


def select_hard_frames(instances, dist_thres=10):
    hard_frames = []
    for i in range(len(instances)):
        mask = instances[i]
        bbox_mask = np.zeros_like(mask)
        bboxes = []
        for region in regionprops(mask):
            if region.area >= 100:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                bbox_mask[minr:maxr, minc:maxc] += 1
                bboxes.append(region.bbox)
            else:
                print('err: ', i)
        assert 2 == len(bboxes)

        # if np.max(bbox_mask) == 1 and 10 <= rect_distance(bboxes[0], bboxes[1]):
        if rect_distance(bboxes[0], bboxes[1]) <= dist_thres:
            hard_frames.append(i)
    return instances[hard_frames], hard_frames


def convert_handannot(annot_dir, img_dir_base, out_dir_base, only_hard_frames=False):
    # Get all sequences
    seqs = utils.io.list_directory(annot_dir)
    # Process each sequence
    for annot_fn in seqs:
        # Init dirs
        img_dir = os.path.splitext(annot_fn)[0].replace(annot_dir, img_dir_base)
        out_dir = img_dir.replace(img_dir_base, out_dir_base)
        # Load annotations and images
        data = utils.io.read(annot_fn)
        imgs = utils.io.read_arrays(utils.io.list_directory(img_dir, sort_key=utils.io.filename_to_number),
                                    utils.io.read)
        print(annot_fn, imgs.shape)

        # Convert annotation to instance masks
        instances = np.zeros((200, 420, 640), dtype=np.uint8)
        for k, v in data['_via_img_metadata'].items():
            fn = os.path.basename(k)
            img_id = int(os.path.splitext(fn)[0])
            for region in v['regions']:
                if region['shape_attributes']['name'] != 'polygon' and region['shape_attributes']['name'] != 'polyline':
                    print('Not a polygon/polyline', region)
                    continue
                xlist, ylist = region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']
                inst_id = region['region_attributes']['instance_id']
                arr = np.asarray(list(zip(xlist, ylist))).ravel().tolist()
                if 0 < len(arr):
                    mask = utils.segmentation.decode_mask(arr, imgs[img_id].shape[:2])
                    instances[img_id] += (int(inst_id)*127*mask).astype(np.uint8)

        # Select hard samples
        if only_hard_frames:
            instances, frame_list = select_hard_frames(instances)
            imgs = imgs[frame_list]
        # Save results
        utils.io.save_images(out_dir, imgs, fn_format='{}{}')
        utils.io.save_images(out_dir.replace('images', 'masks'), instances, fn_format='{}{}')
        utils.io.save_video(out_dir + '.avi', utils.visualization.concat_images([imgs, utils.visualization.grayscale_to_rgb(instances)]))
        # utils.save_video(out_dir + '2.avi', visualization.overlay_mask(imgs, instances))


def convert2coco(img_dir_base, mask_dir_base, out_json):
    # Get all sequences
    seqs = utils.io.list_directory(mask_dir_base, only_dirs=True, full_path=False)
    # Init
    curr_id = 0
    record_list = []
    # Process each sequence
    for seq in seqs:
        # Init dirs
        img_dir = os.path.join(img_dir_base, seq)
        mask_dir = os.path.join(mask_dir_base, seq)
        # Load images and instances
        img_fns = utils.io.list_directory(img_dir, sort_key=utils.io.filename_to_number)
        masks = utils.io.read_arrays(utils.io.list_directory(mask_dir, sort_key=utils.io.filename_to_number),
                                     utils.io.read)
        inst_ids, counts = np.unique(masks[0], return_counts=True)
        inst_ids, counts = inst_ids[1:], counts[1:]
        assert len(img_fns) == len(masks)
        print(seq, inst_ids, counts)
        inst_ids = [127, 254]
        assert len(inst_ids) == 2
        height, width = masks.shape[1:]
        # Process each image
        for fn, mask in zip(img_fns, masks):
            # Init record
            record = {}
            record["file_name"] = fn
            record["image_id"] = curr_id
            curr_id += 1
            record["height"] = height
            record["width"] = width

            record["annotations"] = []
            # Process each instance
            for inst_id in inst_ids:
                instance_mask = (mask == inst_id).astype(np.uint8)
                # Convert instance mask to a coord list and a bbox
                segmentations, area, bbox = a2c.coco_segmentation(instance_mask)
                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": segmentations,
                    "area": area,  # Not needed
                    "keypoints": segmentations[0][:2] + [2] + [0]*6,
                    "category_id": 0,
                    "iscrowd": 0
                }
                record["annotations"].append(obj)
            # Add it annotations
            record_list.append(record)
    # Save annotation
    utils.io.save(out_json, record_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_dir', required=True, help='Directory containing the images.')
    parser.add_argument('-a', '--annot_dir', required=True, help='Directory containing the annotations.')
    parser.add_argument('-o', '--out_dir', required=True, help='Directory to save the dataset.')
    return parser.parse_args()


def run(img_dir, annot_dir, out_dir):
    # Validate parameters
    assert os.path.isdir(img_dir)
    assert os.path.isdir(annot_dir)

    # Initialization
    utils.io.make_directory(out_dir)
    out_img_dir = os.path.join(out_dir, 'images')
    out_mask_dir = os.path.join(out_dir, 'masks')
    out_annot_json = os.path.join(out_dir, 'annotations', 'test.json')

    # Convert dataset
    convert_handannot(annot_dir, img_dir, out_img_dir, only_hard_frames=False)
    convert2coco(out_img_dir, out_mask_dir, out_annot_json)

    return out_img_dir, out_mask_dir, out_annot_json



if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))