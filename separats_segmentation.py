import datetime
import itertools
import multiprocessing
import os
import random
import sys
import traceback

import cv2
import numpy as np
import scipy.spatial
import skimage.measure
import torch
import tqdm

sys.path.append('./external/dexi-hand2')
from model import DexiNed
from dexi_utils import image_normalization
import PartsNEdges4 as segm

sys.path.append('./external/dexi-hand2/sepaRats_notebooks')
import preproc.preproc as prep  # it uses my utils folder

import foreground_segmentation
import utils.io


KEEPFROMSEQ = 0
STARTFRAMETOKEEP = -1

LIMITFRAME = -1  # 3037 #  3037
CROPSEQ_FROM = 15  # 15   #  14

# bg800: object size 800 => kept.
# bg800m: bg800 extended: with maskrcnn bodyparts (basic)
# bg800m2: bg800m extended: mrcnn mask is dilated extremely before intersect with bgdiff+hsv corr fg

########################## remove_background ################################
checked_c1, checked_c2 = 190, 465
diff_thresh = 40
HSV_THRESH = 116
obj_size_limit = 800  # 1000
k = 5  # remove noise kernel size
clahe = 0.8
#############################################################################

GPU_ID = 0
DEVICE = torch.device(f"cuda:{GPU_ID}")


def remove_background(in_img_base_dir, in_bdypts_base_dir,
                      out_img_base_dir, out_fgmask_base_dir, out_bg_base_dir,
                      n_processes=-1):
    seqs = utils.io.list_directory(in_img_base_dir, only_dirs=True, full_path=False)
    # print(seqs)

    in_img_dirs = [os.path.join(in_img_base_dir, seq) for seq in seqs]
    in_bdypts_dirs = [os.path.join(in_bdypts_base_dir, seq) for seq in seqs]
    out_img_dirs = [os.path.join(out_img_base_dir, seq) for seq in seqs]
    out_fgmask_dirs = [os.path.join(out_fgmask_base_dir, seq) for seq in seqs]
    out_bg_paths = [os.path.join(out_bg_base_dir, seq + '_bg.png') for seq in seqs]
    utils.io.make_directories(out_bg_base_dir, *out_img_dirs, *out_fgmask_dirs)

    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, len(seqs))

    if n_processes == 1:
        for (seq_id, in_img_dir, in_bdypts_dir,
             out_bg_path, out_fgmask_dir, out_img_dir) in zip(range(len(seqs)), in_img_dirs, in_bdypts_dirs,
                                                              out_bg_paths, out_fgmask_dirs, out_img_dirs):
            create_background_image(out_bg_path, in_img_dir)
            remove_background_from_sequence(seq_id, in_img_dir, in_bdypts_dir, out_bg_path, out_fgmask_dir, out_img_dir)
    else:
        with multiprocessing.Pool(processes=min(2, n_processes // 4)) as pool:
            with tqdm.tqdm(total=len(seqs)) as pbar:
                for _ in pool.starmap(create_background_image, zip(out_bg_paths, in_img_dirs), chunksize=1):
                    pbar.update()

        with multiprocessing.Pool(processes=n_processes) as pool:
            with tqdm.tqdm(total=len(seqs)) as pbar:
                for _ in pool.starmap(remove_background_from_sequence,
                                      zip(range(len(seqs)), in_img_dirs, in_bdypts_dirs,
                                          out_bg_paths, out_fgmask_dirs, out_img_dirs), chunksize=1):
                    pbar.update()


def create_background_image(out_bg_path, in_img_dir):
    if not os.path.isfile(out_bg_path):
        # print(f"determine_bg {seq_id}")
        determine_background(in_img_dir, out_bg_path, n_samples=300, use_fast=True)


def remove_background_from_sequence_wrapper(params):
    remove_background_from_sequence(*params)
    return params[-1]


def remove_background_from_sequence(seq_id, in_img_dir, in_bdypts_dir,
                                    out_bg_path, out_fgmask_dir, out_img_dir):
    # print(f"start {seq_id}")
    from_frame, to_frames, make_blank = determine_blank_background_interval(seq_id)
    try:
        n_frames = -1
        # print(f"removeBG {seq_id}")
        prep.removeBG(in_bdypts_dir, in_img_dir, out_img_dir, out_fgmask_dir, out_bg_path,
                      diff_thresh, HSV_THRESH, obj_size_limit, k, clahe, False, checked_c1, checked_c2, False,
                      from_frame, n_frames)

        if make_blank:
            bg_img = utils.io.read(out_bg_path)
            blank_img = np.zeros_like(bg_img)
            fns = utils.io.list_directory(in_img_dir, full_path=False, extension='.png')
            if to_frames == -1: to_frames = len(fns)
            for fn in fns[from_frame:to_frames]:
                utils.io.save(os.path.join(out_fgmask_dir, fn), blank_img)
                utils.io.save(os.path.join(out_img_dir, fn), blank_img)
    except Exception as e:
        print('[EXCEPTION]: ', seq_id, e)


def determine_blank_background_interval(seq_id):
    from_frame = 0
    to_frames = -1
    make_blank = False
    if STARTFRAMETOKEEP != -1:
        if seq_id < KEEPFROMSEQ:  # seq001-ban csak a 34.frame-tol kell, seq000 mindere legyen blank
            from_frame = 0
            to_frames = -1
            make_blank = True
        elif seq_id == CROPSEQ_FROM:
            from_frame = 0
            to_frames = STARTFRAMETOKEEP  # till end
            make_blank = True
    if LIMITFRAME != -1:
        if seq_id == CROPSEQ_FROM:  # seq0015
            from_frame = LIMITFRAME
            to_frames = -1
            make_blank = True
        elif seq_id > CROPSEQ_FROM:  # seq0014
            from_frame = 0
            to_frames = -1
            make_blank = True
    return from_frame, to_frames, make_blank


def determine_background(in_img_dir, out_bg_path, n_samples=300, use_fast=True):
    if not use_fast:
        bg_remover = prep.BackgroundRemover(in_img_dir, out_bg_path, has_subdirs=False)
        bg_remover.create_bg_image(n_samples)  # with 300 images
    else:
        fns = utils.io.list_directory(in_img_dir, extension='.png')
        random.seed(0)
        fns = random.sample(fns, min(n_samples, len(fns)))
        imgs = utils.io.read_arrays(fns)
        bg_img, _ = foreground_segmentation.separate_mode(imgs)
        utils.io.save(out_bg_path, bg_img)


def transform(img, cut_center=False, central_box_left_bound=0, central_box_right_bound=420, resize=False):
    if cut_center:
        img = img[:, central_box_left_bound:central_box_right_bound, :]

    if resize:
        img = cv2.resize(img, dsize=(640, 420))

    newimg = np.zeros_like(img)
    img_height = img.shape[0]
    img_width = img.shape[1]
    # Make sure images and labels are divisible by 2^4=16
    if img.shape[0] % 16 != 0 or img.shape[1] % 16 != 0:
        img_height = img.shape[0] if img.shape[0] % 16 == 0 else (
                                                                         (img.shape[0] // 16) + 1) * 16
        img_width = img.shape[1] if img.shape[1] % 16 == 0 else (
                                                                        (img.shape[1] // 16) + 1) * 16

        newimg = cv2.resize(newimg, (img_width, img_height))
        # img = cv2.resize(img, (img_width, img_height))

    newimg[0:img.shape[0], 0:img.shape[1], :] = img
    pad0 = img_height - img.shape[0]
    pad1 = img_width - img.shape[1]
    newimg = np.array(newimg, dtype=np.float32)
    newimg = newimg.transpose((2, 0, 1))
    # img = torch.from_numpy(img.copy()).float()
    newimg = torch.from_numpy(newimg.copy()).float()

    img = newimg
    return img, (pad0, pad1)


def load_model(checkpoint_path):
    model = DexiNed().to(DEVICE)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    # Put model in evaluation mode
    model.eval()
    return model


def predict_with_model(model, mean, image_dir, image_fname, cut_center=False, resize=False):
    img = utils.io.read(os.path.join(image_dir, image_fname))
    if resize:
        img = cv2.resize(img, dsize=(640, 420))

    img = np.array(img, dtype=np.float32)
    img -= np.array(mean).astype(np.float)

    img, padding = transform(img, cut_center=cut_center)

    with torch.no_grad():
        images = img[None, :, :, :].to(DEVICE)
        preds = model(images)

    edge_maps = []
    for i in preds:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    edge_maps = np.array(edge_maps)
    edge_maps = np.squeeze(edge_maps)

    return edge_maps, padding


def display_layer_outputs(edge_maps, output_dir, output_fname, padding):
    # plt.figure(figsize=(20, 50))
    for i in range(edge_maps.shape[0]):
        tmp_img = edge_maps[i].copy()

        h = tmp_img.shape[0] - padding[0]
        w = tmp_img.shape[1] - padding[1]
        tmp_img_orig = np.zeros((h, w))
        tmp_img_orig = tmp_img[0:h, 0:w]
        tmp_img = tmp_img_orig

        tmp_img = np.uint8(image_normalization(tmp_img))
        # if 'sepaRats' in modelname:
        tmp_img = cv2.bitwise_not(tmp_img)

        if i == 6:
            utils.io.save(os.path.join(output_dir, output_fname), tmp_img)


def calculate_mean(img_dir, c=0.8, resize=False, has_subdirs=False):
    print('calculate image mean')
    max_number_of_files_to_read = 5000
    ch_sums = np.zeros(3)  # sums across 3 channels
    image_counter = 0

    if has_subdirs:
        subdirs = os.listdir(img_dir)
    else:
        subdirs = ['']

    for s in range(len(subdirs)):
        files = os.listdir(os.path.join(img_dir, subdirs[s]))
        for f in range(len(files)):
            image = utils.io.read(os.path.join(img_dir, subdirs[s], files[f]))

            if resize:
                image = cv2.resize(image, dsize=(640, 420))

            ch_sums += np.mean(image, axis=(0, 1))

            image_counter += 1
            if image_counter > max_number_of_files_to_read:
                break

    means = ch_sums / image_counter
    return means


def run_dexined(out_pred_base_dir, out_img_base_dir, do_calculate_mean=False,
                model_path='./models/separats/dexined/4_model.pth',
                n_processes=-1):
    if do_calculate_mean:
        means = calculate_mean(out_img_base_dir, 0.8, has_subdirs=False)
    else:
        means = [15.459616304278207, 17.138717755106207, 17.443830066384184]

    seqs = utils.io.list_directory(out_img_base_dir, only_dirs=True, full_path=False)
    out_img_dirs = [os.path.join(out_img_base_dir, seq) for seq in seqs]
    out_pred_dirs = [os.path.join(out_pred_base_dir, seq) for seq in seqs]
    utils.io.make_directories(*out_pred_dirs)

    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, len(seqs))

    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm.tqdm(total=len(seqs)) as pbar:
            for _ in pool.starmap(detect_edges,
                                  zip(itertools.repeat(model_path), itertools.repeat(means),
                                      out_img_dirs, out_pred_dirs), chunksize=1):
                pbar.update()



def detect_edges(model_path, means, out_img_dir, out_pred_dir):
    model = load_model(model_path)
    fns = utils.io.list_directory(out_img_dir, extension='.png', full_path=False)
    for fn in fns:
        try:
            edge_maps, padding = predict_with_model(model, means, out_img_dir, fn, cut_center=False)
            display_layer_outputs(edge_maps, out_pred_dir, fn, padding)
        except Exception as e:
            print('[EXCEPTION]', fn, e)


def FoI(s, f):  # frame of interest
    is_frame_of_interest = True

    if LIMITFRAME != -1:
        if s == CROPSEQ_FROM and f >= LIMITFRAME:  # seq0014
            is_frame_of_interest = False
        if s > CROPSEQ_FROM:
            is_frame_of_interest = False
    if STARTFRAMETOKEEP != -1:
        if s == KEEPFROMSEQ and f <= STARTFRAMETOKEEP:  # seq0014
            is_frame_of_interest = False
        if s < KEEPFROMSEQ:
            is_frame_of_interest = False
    return is_frame_of_interest


def getmask_fromRGB(img):
    yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y_img = yuv_img[:, :, 0]
    ret, thresh1 = cv2.threshold(y_img, 116, 255, cv2.THRESH_BINARY)  # set threshold to 116 -- empiric value :)
    rats = remove_small_objects(thresh1)

    rats[:, 0:checked_c1] = 0
    rats[:, checked_c2:-1] = 0
    rats = cv2.dilate(rats, np.ones((3, 3)))
    rats = cv2.erode(rats, np.ones((3, 3)))
    inner_mask = rats
    return inner_mask


def prep_edge(rgb_img, edge_img, ksize=13):
    roi_rgb = rgb_img.copy()
    roi_edge = edge_img.copy()
    roi_rgb *= 0
    roi_edge *= 0
    roi_rgb[:, checked_c1:checked_c2, :] = rgb_img[:, checked_c1:checked_c2, :]
    roi_edge[:, checked_c1:checked_c2] = edge_img[:, checked_c1:checked_c2]

    # ret,roi_edge = cv2.threshold(roi_edge,40,255,cv2.THRESH_BINARY)
    ret, roi_edge = cv2.threshold(roi_edge, 20, 255, cv2.THRESH_BINARY)

    yuv_img = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2YUV)
    # get channel Y
    y_img = np.zeros([yuv_img.shape[0], yuv_img.shape[1]])
    y_img = yuv_img[:, :, 0]
    # set threshold to 116
    ret, thresh1 = cv2.threshold(y_img, 116, 255, cv2.THRESH_BINARY)
    rats = remove_small_objects(thresh1)

    binary_ratmaskimg = cv2.dilate(rats, np.ones((ksize, ksize)))

    result = cv2.bitwise_and(roi_edge.astype(np.uint8), roi_edge.astype(np.uint8), mask=binary_ratmaskimg)
    return result


def remove_small_objects(img, min_size=150):  # 30 is big enough
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img2 = img
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            img2[output == i + 1] = 0

    return img2


def get_in_mask(img, roi_edge):  # , gtedge):
    rats = roi_edge
    cv2.floodFill(rats, None, (1, 1), 255)
    rats = 255 - rats
    rats = cv2.bitwise_and(rats, getmask_fromRGB(img))
    return rats


def split_data_into_separated_and_occluded(out_pred_base_dir, out_base_dir, out_img_base_dir, out_fgmask_base_dir):
    sep_list_path = os.path.join(out_base_dir, 'sep_list.txt')
    occ_list_path = os.path.join(out_base_dir, 'occ_list.txt')

    if os.path.isfile(sep_list_path) and os.path.getsize(sep_list_path) > 0:
        print('Load lists from file.')
        separated_fns = [line.strip() for line in utils.io.read(sep_list_path)]
        occluded_fns = [line.strip() for line in utils.io.read(occ_list_path)]
    else:
        separated_fns = []
        occluded_fns = []

        seqs = utils.io.list_directory(out_pred_base_dir, only_dirs=True, full_path=False)
        for seq_id, seq in enumerate(seqs):
            curr_gappy_dir = os.path.join(out_pred_base_dir, seq)
            curr_img_dir = os.path.join(out_img_base_dir, seq)
            fgmaskpath = os.path.join(out_fgmask_base_dir, seq)

            fns = utils.io.list_directory(curr_img_dir, extension='.png', full_path=False)

            for fn_id, fn in enumerate(fns):
                if FoI(seq_id, fn_id):
                    try:
                        rgbimg = utils.io.read(os.path.join(curr_img_dir, fn))
                        edgeimg = utils.io.read(os.path.join(curr_gappy_dir, fn))

                        fgmask = utils.io.read(os.path.join(fgmaskpath, fn))

                        labeled_fgmask = skimage.measure.label(fgmask)
                        props_fgmask = skimage.measure.regionprops(labeled_fgmask)
                        numOfobj_fgm = 0

                        for p in range(len(props_fgmask)):
                            # ez csak elovigyazat vsz eleg lenne nezni a len()>1, de nehogy zaj felhuzza
                            numOfobj_fgm = numOfobj_fgm + 1 if props_fgmask[p].area > 100 else numOfobj_fgm + 0

                        if numOfobj_fgm > 1:
                            separated_fns.append(seq + '/' + fn)
                        else:
                            # éldetection szétszedi-e
                            edgeimg = prep_edge(rgbimg, edgeimg)
                            # bodyedge = edgeimg.copy()
                            edgeimg = cv2.dilate(edgeimg, np.ones((5, 5)))
                            edgeimg = cv2.erode(edgeimg, np.ones((5, 5)))
                            bodyedge = edgeimg.copy()

                            ret, edgeimg = cv2.threshold(edgeimg, 110, 255, cv2.THRESH_BINARY)
                            joint_bodies_mask = get_in_mask(rgbimg, edgeimg)  # without tails

                            labelled_pre = skimage.measure.label(joint_bodies_mask * 255)
                            props_pre = skimage.measure.regionprops(labelled_pre)

                            numOfobj_pre = 0
                            for p in range(len(props_pre)):
                                # print('area : ', props_pre[p].area)
                                numOfobj_pre = numOfobj_pre + 1 if props_pre[p].area > 600 else numOfobj_pre + 0

                            if numOfobj_pre > 1:
                                separated_fns.append(seq + '/' + fn)

                            else:
                                occluded_fns.append(seq + '/' + fn)
                    except Exception as e:
                        print('[EXCEPTION]', os.path.join(curr_img_dir, fn), e)
        utils.io.save(sep_list_path, separated_fns)
        utils.io.save(occ_list_path, occluded_fns)

    print('#sep/#occ frames for ', out_pred_base_dir)
    print('- sep: ', len(separated_fns))
    print('- occ', len(occluded_fns))
    return separated_fns


def generate_maskG(out_pred_base_dir, out_fgmask_base_dir, out_img_base_dir,
                   out_gappy_masks_base_dir, out_tmp_edges_base_dir, out_tmp_imgs_base_dir,
                   occ_list_path):
    occluded_fns = utils.io.read(occ_list_path)
    occluded_fns = set(map(lambda fn: fn.strip(), occluded_fns))
    seqs = utils.io.list_directory(out_pred_base_dir, only_dirs=True, full_path=False)

    for seq_id, seq in enumerate(seqs):
        edge_dir = os.path.join(out_pred_base_dir, seq)  # incomplete edges
        mask_dir = os.path.join(out_fgmask_base_dir, seq)
        nobg_dir = os.path.join(out_img_base_dir, seq)

        # outdirs
        maskG_dir = os.path.join(out_gappy_masks_base_dir, seq)
        edge_tmpOUT = os.path.join(out_tmp_edges_base_dir, seq)
        imgs_tmpOUT = os.path.join(out_tmp_imgs_base_dir, seq)

        utils.io.make_directories(edge_tmpOUT, imgs_tmpOUT)

        fns = utils.io.list_directory(mask_dir, extension='.png', full_path=False,
                                      fn_constraint=lambda fn: os.path.join(seq, fn) in occluded_fns)
        # TODO: use occluded_fns instead of iterating over the whole dataset...
        print(' occ: ', len(fns))

        outSUBdir = maskG_dir
        os.makedirs(outSUBdir, exist_ok=True)

        for fn in fns:
            num = utils.io.filename_to_number(fn)
            if FoI(seq_id, num):
                # gappy: black bg, white edges
                gappy = utils.io.read(os.path.join(edge_dir, fn))
                nobgimg = utils.io.read(os.path.join(nobg_dir, fn))

                utils.io.save(os.path.join(imgs_tmpOUT, fn), nobgimg)
                utils.io.save(os.path.join(edge_tmpOUT, fn), gappy)

                ret, gappyt = cv2.threshold(gappy, 44, 255, cv2.THRESH_BINARY)
                gappy = 255 - gappyt

                # get inner mask:
                maskgray = utils.io.read(os.path.join(mask_dir, fn))

                if np.max(maskgray) == 0:
                    pass
                else:
                    try:
                        ret, mask = cv2.threshold(maskgray, 10, 1, cv2.THRESH_BINARY)
                        mask = cv2.erode(mask, np.ones((13, 13)))  # inner mask

                        # edge info beneath the inner mask:
                        gappy_masked = (255 - gappy) * mask
                        ret, gappymasked = cv2.threshold(gappy_masked, 10, 1, cv2.THRESH_BINARY)

                        # inner mask correction with edge info:
                        mask = mask - gappymasked
                        mask = mask * 255

                        utils.io.save(os.path.join(outSUBdir, fn), mask)
                    except Exception as e:
                        print('[EXCEPTION]', os.path.join(mask_dir, fn), e)


def edge_completion(GLOBAL_TMP_EDGES, GLOBAL_TMP_IMGS, GLOBAL_gappymasks_dir, GLOBAL_fgmasks_dir,
                    GLOBAL_edgecompl_stitch, GLOBAL_edgecompl_out, GLOBAL_base_dir, GLOBAL_modelID,
                    n_processes=-1):
    ## batch output generation for MULTIPLE EDGECONNECT models ON MULTIPLE DATADIRS
    # for Noblur_repr full dexined only

    edge_tmpOUT = GLOBAL_TMP_EDGES
    imgs_tmpOUT = GLOBAL_TMP_IMGS

    edgeModel = 'nlb_2loss_rC'
    ep = 4

    print('========================== MODEL:: ', edgeModel + '__[' + str(ep) + ']  ==========================')
    configyaml = 'config.yml'

    seqs = utils.io.list_directory(imgs_tmpOUT, only_dirs=True, full_path=False)
    chp = './models/separats/edgeconnect/' + edgeModel
    print(chp)

    commands = []
    for s in range(len(seqs)):
        print(s, ':: ', seqs[s])
        print(s + 1, '/', len(seqs), end='\r')
        curr_im_dir = os.path.join(imgs_tmpOUT, seqs[s])  # GLOBAL_noBGin_dir
        curr_gappy_dir = os.path.join(edge_tmpOUT, seqs[s])  # GLOBAL_dxndoutdir

        curr_mG_dir = os.path.join(GLOBAL_gappymasks_dir, seqs[s])
        curr_m_dir = os.path.join(GLOBAL_fgmasks_dir, seqs[s])

        if len(os.listdir(curr_im_dir)) < 1:
            pass
        else:
            curr_out_dir_stitched = os.path.join(GLOBAL_edgecompl_stitch, seqs[s])
            curr_out_m2_dir = os.path.join(GLOBAL_edgecompl_out, seqs[s])

            curr_out_m1_dir = os.path.join(GLOBAL_base_dir + '_unused', 'edges_m1' + GLOBAL_modelID, seqs[s])
            curr_out_enh_m1_dir = os.path.join(GLOBAL_base_dir + '_unused', 'edges_enh1' + GLOBAL_modelID, seqs[s])
            curr_out_enh_m2_dir = os.path.join(GLOBAL_base_dir + '_unused', 'edges_enh2' + GLOBAL_modelID, seqs[s])
            curr_segm1_out_dir = os.path.join(GLOBAL_base_dir + '_unused', 'segm_1' + GLOBAL_modelID, seqs[s])
            curr_segm2_out_dir = os.path.join(GLOBAL_base_dir + '_unused', 'segm_2' + GLOBAL_modelID, seqs[s])

            commands.append(f"python ./external/dexi-hand2/src_edgeconnect/test.py --model 1 --checkpoints '{chp}' "
                            f"--input '{curr_im_dir}' --mask '{curr_mG_dir}' --edge '{curr_gappy_dir}' "
                            f"--segm '{curr_m_dir}' --output_stitched '{curr_out_dir_stitched}' "
                            f"--output_merge1 '{curr_out_m1_dir}' --output_merge2 '{curr_out_m2_dir}' "
                            f"--output_enh_m1 '{curr_out_enh_m1_dir}' --output_enh_m2 '{curr_out_enh_m2_dir}' "
                            f"--segm_res1_path '{curr_segm1_out_dir}' --segm_res2_path '{curr_segm2_out_dir}' "
                            f"--config '{configyaml}' --enhance True --epoch {ep} --set1GPU {GPU_ID} --stitched True")

    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, len(commands))

    with multiprocessing.Pool(min(n_processes, len(commands))) as pool:
        with tqdm.tqdm(total=len(commands)) as pbar:
            for _ in pool.imap_unordered(run_bash_command, commands):
                pbar.update()


def run_bash_command(command):
    # print(command)
    os.system(command)


def keep_2_largest_nonoverlap_bodies(bodies, noise):
    bodies_selected = []
    bodies_selected.append(bodies[0])

    for b in range(1, len(bodies)):
        overlap = 0
        for bs in range(len(bodies_selected)):
            overlap += np.sum(cv2.bitwise_and(bodies_selected[bs], bodies[b]))

        if overlap < 1000:
            bodies_selected.append(bodies[b])
    bodies = bodies_selected

    if (len(bodies) > 2):
        mini = 20000
        index = -1
        for b in range(len(bodies)):
            if int(mini) > int(np.sum(bodies[b])):
                mini = np.sum(bodies[b]).copy()
                index = b
        noise = bodies.pop(index)

    return bodies, noise


def disjoin_bodies(bodies):
    intersectr, intersectc = np.where(np.logical_and(bodies[0] == bodies[1], bodies[0] != 0))

    # ha átefed a két body
    if len(intersectr) > 0:
        bds = np.zeros((bodies[0].shape[0], bodies[0].shape[1], 2), np.uint8)
        bds[..., 0] = bodies[0]
        bds[..., 1] = bodies[1]

        rr, cc = np.where(bodies[0] == 1)
        rro, cco = np.where(bodies[1] == 1)

        for k in range(len(intersectr)):
            a = [intersectr[k], intersectc[k]]
            b1 = [np.mean(rr).astype(int), np.mean(cc).astype(int)]
            b2 = [np.mean(rro).astype(int), np.mean(cco).astype(int)]
            dst1 = scipy.spatial.distance.euclidean(a, b1)
            dst2 = scipy.spatial.distance.euclidean(a, b2)

            if dst1 < dst2:
                bds[intersectr[k], intersectc[k], 0] = 1
                bds[intersectr[k], intersectc[k], 1] = 0
            else:
                bds[intersectr[k], intersectc[k], 1] = 1
                bds[intersectr[k], intersectc[k], 0] = 0

        bodies[0] = bds[..., 0]
        bodies[1] = bds[..., 1]

    bodies_0 = bodies[0].copy()  # [r1:r2, c1:c2]
    bodies_1 = bodies[1].copy()  # [r1:r2, c1:c2]

    t0, _ = np.where(bodies_0 != 0)  # where body 0th
    t1, _ = np.where(bodies_1 != 0)  # where body 1th

    # bigger body is bodies[0]
    bodies[0] = bodies_0 if len(t0) > len(t1) else bodies_1
    bodies[1] = bodies_1 if len(t0) > len(t1) else bodies_0

    return bodies


def is_separated_frame(fgmask, minsize=200):
    noNoise = remove_small_objects(fgmask, minsize)  # leszedjük ami nyilvánvalóan nem patkány
    fg_regions = skimage.measure.label(noNoise, connectivity=1)
    fg_props = skimage.measure.regionprops(fg_regions)

    if len(fg_props) < 2:
        return False
    else:
        return True


def create_segmentation(GLOBAL_input_dir, GLOBAL_fgmasks_dir, GLOBAL_bdypts_dir, GLOBAL_edgecompl_out,
                        GLOBAL_segm_out, GLOBAL_dxndoutdir,
                        occ_list_path, n_processes=-1):
    print('parts:: ', GLOBAL_bdypts_dir)
    seqs = utils.io.list_directory(GLOBAL_input_dir, only_dirs=True, full_path=False)

    occluded_fns = set(map(lambda fn: fn.strip(), utils.io.read(occ_list_path)))

    n_finished = 0
    n_skip = 0
    for seq_id, seq in enumerate(seqs):  # 1,2):#
        if seq_id < n_finished:
            continue
        print(seq, ':: ', seq_id + 1, '/', len(seqs))  # , end='\r')
        # input data directories:
        base_input_dir = os.path.join(GLOBAL_input_dir, seq)
        base_fgmask_dir = os.path.join(GLOBAL_fgmasks_dir, seq)
        base_bdypts_dir = os.path.join(GLOBAL_bdypts_dir, seq)
        base_edge_m2_dir = os.path.join(GLOBAL_edgecompl_out, seq)

        out_segm_dir = os.path.join(GLOBAL_segm_out, seq)
        utils.io.make_directories(out_segm_dir)

        fns = utils.io.list_directory(base_input_dir, extension='.png', full_path=False)

        if seq_id == n_finished:
            fns = fns[n_skip:]
        out_fns = [os.path.join(base_edge_m2_dir, fn) if os.path.join(seq, fn) in occluded_fns
                   else os.path.join(GLOBAL_dxndoutdir, seq, fn) for fn in fns]

        '''
        for fn, path_output_m2 in tqdm.tqdm(zip(fns, out_fns)):
            segm2 = determine_segmentation(seq_id, fn, path_output_m2, base_bdypts_dir, base_fgmask_dir)

            path_segm_bdypts2 = os.path.join(out_segm_dir, fn)
            utils.io.save(path_segm_bdypts2, segm2)
        '''

        if n_processes < 1:
            n_processes = multiprocessing.cpu_count()

        with multiprocessing.Pool(processes=n_processes) as pool:
            for fn_id, segm2 in enumerate(pool.starmap(determine_segmentation,
                                                       tqdm.tqdm(zip(itertools.repeat(seq_id), fns, out_fns,
                                                                     itertools.repeat(base_bdypts_dir),
                                                                     itertools.repeat(base_fgmask_dir)),
                                                                 total=len(fns)),
                                                       chunksize=1)):
                path_segm_bdypts2 = os.path.join(out_segm_dir, fns[fn_id])
                utils.io.save(path_segm_bdypts2, segm2)
        print('\n')


def determine_segmentation(seq_id, fn, path_output_m2, base_bdypts_dir, base_fgmask_dir):
    segm2 = np.zeros((420, 640))
    fn_number = utils.io.filename_to_number(fn)
    if FoI(seq_id, fn_number):
        path_parts_json = os.path.join(base_bdypts_dir, 'annot', fn.replace('.png', '.json'))
        path_parts_imgs = os.path.join(base_bdypts_dir, 'annot', fn)
        path_fgmask = os.path.join(base_fgmask_dir, fn)

        fgmask = utils.io.read(path_fgmask)
        segm2 = np.zeros_like(fgmask)
        if np.sum(fgmask) > 0:
            data = utils.io.read(path_parts_json)
            parts_orig = utils.io.read(path_parts_imgs)
            output_m2 = utils.io.read(path_output_m2)

            heads, bodies, tails, _ = segm.get_bodypartmasks(data, parts_orig)

            # ha tobb mint ketto van, a diszjunktak kozul tartsa meg a ket legnagyobbat
            noise = None
            if len(bodies) > 2:
                bodies, noise = keep_2_largest_nonoverlap_bodies(bodies, noise)
            if len(bodies) > 1:
                bodies = disjoin_bodies(bodies)
            elif len(bodies) == 1:
                bodies.append(None)
            else:
                bodies.append(None)
                bodies.append(None)

            if len(tails) < 1: tails.append(None)
            if len(tails) < 2: tails.append(None)

            if len(heads) > 1:
                heads[0] = (heads[0] * 255).astype(int)
                heads[1] = (heads[1] * 255).astype(int)
            elif len(heads) == 1:
                heads[0] = (heads[0] * 255).astype(int)
                heads.append(None)  # heads[1][r1:r2, c1:c2]
            else:
                heads.append(None)
                heads.append(None)

            r = 1
            out2, segm2, r = segm.do_enhance3(r, output_m2, fgmask, 60,
                                              heads[0], heads[1],
                                              bodies[0], bodies[1], sepImg=is_separated_frame(fgmask, 400),
                                              doplots=False,
                                              tails0=tails[0], tails1=tails[1])

            if len(np.unique(segm2)) > 4:  # 0,127,254 + 40for head
                print('somehow grayscale result @ ', seq_id, '/', fn)
    return segm2


def run(input_dir=None, n_processes=-1, batch_size=4):
    GLOBAL_input_dirs = [
        #"/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_2",
        #"/media/hdd2/lkopi/datasets/Social behavior software experiments/1-Autism study (VPA-model)/VPA control/1-Autism study_control animals_6",
        #"/media/hdd2/lkopi/datasets/Social behavior software experiments/2-Stress study (RT)/RT animals/2_stress study_stressed animals_1",
        #"/media/hdd2/lkopi/datasets/Social behavior software experiments/2-Stress study (RT)/RT animals/2_stress study_stressed animals_2",
        #"/media/hdd2/lkopi/datasets/Social behavior software experiments/2-Stress study (RT)/RT animals/2_stress study_stressed animals_3",
        #"/media/hdd2/lkopi/datasets/Social behavior software experiments/2-Stress study (RT)/RT animals/2_stress study_stressed animals_4",
        #"/media/hdd2/lkopi/datasets/Social behavior software experiments/3-Anxiolytic study (Diazepam injection)/Diazepam injection-20201014/3-diazepam anxiolyitic study_sedated animals-1",
        #"/media/hdd2/lkopi/datasets/Social behavior software experiments/3-Anxiolytic study (Diazepam injection)/Diazepam injection-20201014/3-diazepam anxiolyitic study_sedated animals-2",
        "/media/hdd2/lkopi/datasets/Social behavior software experiments/3-Anxiolytic study (Diazepam injection)/Diazepam injection-20201014/3-diazepam anxiolyitic study_sedated animals-3",
        "/media/hdd2/lkopi/datasets/Social behavior software experiments/3-Anxiolytic study (Diazepam injection)/Diazepam injection-20201014/3-diazepam anxiolyitic study_sedated animals-4",
    ]
    if input_dir is not None:
        GLOBAL_input_dirs = [input_dir]

    for GLOBAL_input_dir in GLOBAL_input_dirs:
        GLOBAL_bdypts_dir = f"{GLOBAL_input_dir}_00012_part/eval_results"
        GLOBAL_base_dir = f"{GLOBAL_input_dir}_separats"

        GLOBAL_EXT = ''
        GLOBAL_modelID = ''  # '_rs' # ''

        GLOBAL_noBGin_dir = GLOBAL_base_dir + '/images_noBG' + GLOBAL_EXT
        GLOBAL_fgmasks_dir = GLOBAL_base_dir + '/fgmasks' + GLOBAL_EXT

        GLOBAL_bgfiles_dir = GLOBAL_base_dir + '/bg_files'
        os.makedirs(GLOBAL_bgfiles_dir, exist_ok=True)
        GLOBAL_dxndoutdir = GLOBAL_base_dir + '/dxnd_out_sepaRats_col3r_ep14'
        # GLOBAL_base_dir + '/res_dxnd_'# write dexined results to this directory

        GLOBAL_gappymasks_dir = GLOBAL_base_dir + '/masksG' + GLOBAL_modelID
        GLOBAL_TMP_EDGES = GLOBAL_base_dir + '/tmp_edges' + GLOBAL_modelID
        GLOBAL_TMP_IMGS = GLOBAL_base_dir + '/tmp_imgs' + GLOBAL_modelID

        GLOBAL_edgecompl_out = GLOBAL_base_dir + '/result_edgeCompl' + GLOBAL_modelID
        GLOBAL_edgecompl_stitch = GLOBAL_base_dir + '/result_edgeCompl_stitch' + GLOBAL_modelID
        GLOBAL_segm_out = GLOBAL_base_dir + '/result_segm_v4.4_col3ep14'

        try:
            print('[remove_background]')  # 10+20min
            remove_background(GLOBAL_input_dir, GLOBAL_bdypts_dir,
                              GLOBAL_noBGin_dir, GLOBAL_fgmasks_dir, GLOBAL_bgfiles_dir,
                              n_processes=n_processes)
            print('[run_dexined]')
            run_dexined(GLOBAL_dxndoutdir, GLOBAL_noBGin_dir, n_processes=batch_size)  # 6)
            print('[split_data_into_separated_and_occluded]')
            split_data_into_separated_and_occluded(GLOBAL_dxndoutdir, GLOBAL_base_dir,
                                                   GLOBAL_noBGin_dir, GLOBAL_fgmasks_dir)
            print('[generate_maskG]')
            generate_maskG(GLOBAL_dxndoutdir, GLOBAL_fgmasks_dir, GLOBAL_noBGin_dir,
                           GLOBAL_gappymasks_dir, GLOBAL_TMP_EDGES, GLOBAL_TMP_IMGS,
                           occ_list_path=os.path.join(GLOBAL_base_dir, 'occ_list.txt'))
            print('[edge_completion]')
            edge_completion(GLOBAL_TMP_EDGES, GLOBAL_TMP_IMGS, GLOBAL_gappymasks_dir, GLOBAL_fgmasks_dir,
                            GLOBAL_edgecompl_stitch, GLOBAL_edgecompl_out, GLOBAL_base_dir, GLOBAL_modelID,
                            n_processes=batch_size)  # 4)
            print('[create_segmentation]')  # 128=16*8min
            create_segmentation(GLOBAL_input_dir, GLOBAL_fgmasks_dir, GLOBAL_bdypts_dir, GLOBAL_edgecompl_out,
                                GLOBAL_segm_out, GLOBAL_dxndoutdir,
                                occ_list_path=os.path.join(GLOBAL_base_dir, 'occ_list.txt'),
                                n_processes=n_processes)
        except Exception as e:
            print('[EXCEPTION in run()]', e)
            traceback.print_exc()

            with open('separats_log_gpu0.txt', 'a') as f:
                f.write(f'[{datetime.datetime.now()}] {GLOBAL_input_dir} {e}\n')
                f.write(traceback.format_exc())


if __name__ == '__main__':
    run(n_processes=8)
