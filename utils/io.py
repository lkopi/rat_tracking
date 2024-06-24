#!/usr/bin/env python
# coding: utf-8

from functools import partial
import json
import multiprocessing
import os
import pickle
import shutil

import cv2
import mmcv
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import pandas as pd
import scipy.io
from tqdm import tqdm

from utils.flow import read_flow
from utils.segmentation import crop_center


#################################
# Functions to list directories #
#################################

def filename_to_number(fn):
    """Converts the given filename to a number.

    Parameters
    ----------
    fn : str
        Filename without path.

    Returns
    -------
    int
        Filename converted to number
    """
    return int(os.path.splitext(fn)[0])


def is_int(s):
    """Returns True if the given string represents an integer.

    Parameters
    ----------
    s : str
        Text that may represent an integer.

    Returns
    -------
    bool
        Returns True if the given string represents an integer.
    """
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()


def list_directory(dir_name, extension=None, sort_key=None, reverse=False,
                   full_path=True, fn_constraint=None, only_dirs=False):
    """List all files with the given extension in the specified directory.

    Parameters
    ----------
    dir_name : str
        Name of the directory.
    extension : str, optional
        Filename suffix, by default None.
    sort_key : function, optional
        A function of one argument that is used to compare filenames. It takes a string and returns an integer.
    reverse : bool, optional
        If true, then sorted in descending order.
    full_path : bool, optional
        Whether to return filenames with the given directory name or not.
    fn_constraint : function, optional
        A boolean function of one argument that is used to filter out invalid filenames.
    only_dirs : str, optional
        Only list directories, by default False.

    Returns
    -------
    list of str
        List of file locations
    """
    is_valid = lambda fn: True
    if extension is None and not only_dirs:
        is_valid = os.path.isfile
    elif extension is not None:
        is_valid = lambda fn: fn.endswith(extension)
    elif only_dirs:
        is_valid = os.path.isdir
    if fn_constraint is None:
        fn_constraint = lambda fn: True

    fns = [fn for fn in os.listdir(dir_name) if is_valid(os.path.join(dir_name, fn)) and fn_constraint(fn)]
    fns.sort(key=sort_key, reverse=reverse)
    if full_path:
        fns = [os.path.join(dir_name, fn) for fn in fns]
    return fns


###########################
# Functions to read files #
###########################

def read_json(json_fn):
    with open(json_fn) as f:
        data = json.load(f)
    return data


def read_video_length(video):
    assert os.path.isfile(video)

    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return length


def read_frames_from_video(video, start_pos=0, n_frames=None, shape=None, crop_rcrc=None):
    assert os.path.isfile(video)
    assert isinstance(start_pos, int) and 0 <= start_pos
    assert n_frames is None or (isinstance(n_frames, int) and 0 < n_frames)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames is None:
        n_frames = length
    if start_pos + n_frames >= length:
        n_frames = length - start_pos
    # assert start_pos <= length - n_frames

    if shape is None:
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        h, w = shape[:2]
    cap.set(1, start_pos)
    frames = np.empty((n_frames, h, w, 3), dtype=np.uint8)
    i = 0
    while cap.isOpened() and i < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if crop_rcrc is not None:
            frame = frame[crop_rcrc[0]:crop_rcrc[2], crop_rcrc[1]:crop_rcrc[3]]
        if shape is not None:
            frame = cv2.resize(frame, (w, h))
        frames[i] = frame
        i += 1
    cap.release()
    return frames, fps


def read(path, img_flag=cv2.IMREAD_UNCHANGED, use_mmcv=False, use_numpy=False):
    """
    Loads the content of a file. It is mainly a convenience function to
    avoid adding the ``open()`` contexts. File type detection is based on extensions.
    Can handle the following types:

    - .txt: text files, result is a list of strings ending whitespace removed
    """
    assert os.path.isfile(path), path

    if path.endswith(('.txt', '.csv')):
        if use_numpy:
            return np.genfromtxt(path, delimiter=',')
        with open(path, 'r') as f:
            return f.readlines()
    if path.endswith('xlsx'):
        data = pd.read_excel(path)
        if use_numpy:
            data = data.to_numpy()
        return data
    elif path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    elif not use_mmcv and path.endswith(('.png', '.jpg')):
        # return cv2.imread(path, img_flag)
        # source: https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), img_flag)
    elif use_mmcv and path.endswith(('.png', '.jpg')):
        return mmcv.flowread(path, quantize=True, concat_axis=0)
    elif path.endswith('.avi'):
        return read_frames_from_video(path)[0]
    elif path.endswith('.flo'):
        return read_flow(path)
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith('.mat'):
        return scipy.io.loadmat(path)
    else:
        raise NotImplementedError('Unknown extension: ' + os.path.splitext(path)[1])


def read_files(fns, reader=read):
    """Read the given list of files using the specified reader function.

    Parameters
    ----------
    fns : list of str
        List of filenames.
    reader : callable ``reader(str)``, optional
        Function to read a file into a numpy array.
    Returns
    -------
    list of objects
        A list containing all the read files.
    """
    assert 0 < len(fns)
    data = []
    for idx, fn in enumerate(fns):
        data.append(reader(fn))
    return data


def read_arrays(fns, reader=read, shape=None, dtype=None, use_cropping=False, **kwargs):
    """Read the given list of files using the specified reader function.

    Parameters
    ----------
    fns : list of str
        List of filenames.
    reader : callable ``reader(str)``, optional
        Function to read a file into a numpy array.
    shape : tuple of int, optional
        Expected shape of the read files, if not specified use the shape of the first file, by default None.
    dtype : data-type, optional
        Data type of the read files, if not specified use the type of the first file, by default None.
    use_cropping : bool, optional
        If true, crop the arrays if they don't match with the expected shape, else resize the image, by default False.

    Returns
    -------
    np.ndarray
        A matrix containing all the read files.
    """
    assert 0 < len(fns)

    tmp = reader(fns[0], **kwargs)
    if shape is None:
        shape = (len(fns), *tmp.shape)
        expected_shape = tmp.shape
    else:
        assert shape[0] == len(fns)
        expected_shape = shape[1:]
    height, width = expected_shape[:2]
    if dtype is None:
        dtype = tmp.dtype

    data = np.empty(shape, dtype=dtype)
    for idx, fn in enumerate(fns):
        tmp = reader(fn, **kwargs)
        if expected_shape != tmp.shape:
            if not use_cropping:
                tmp = cv2.resize(tmp, (width, height))
            else:
                tmp = crop_center(tmp, cropped_height=height, cropped_width=width)
        data[idx] = tmp

    return data


def read_arrays_parallel(fns, chunksize=100, n_processes=-1):
    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, max(1, len(fns)//chunksize))
    if n_processes == 1:
        return read_arrays(fns)

    arr = read(fns[0])
    arr_shape = (len(fns), *arr.shape)
    out = np.empty(arr_shape, dtype=arr.dtype)
    del arr
    # TODO split fns in list of lists
    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm(total=len(fns)) as pbar:
            for i, arr in enumerate(pool.imap(read_arrays, ([fn] for fn in fns), chunksize=chunksize)):
                out[i] = arr
                pbar.update()
    return out


def read_files_parallel(fns, chunksize=100, n_processes=-1):
    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, max(1, len(fns)//chunksize))
    if n_processes == 1:
        return read_files(fns)

    assert 0 < len(fns)
    out = [None]*len(fns)

    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm(total=len(fns)) as pbar:
            for i, arr in enumerate(pool.imap(read, fns, chunksize=chunksize)):
                out[i] = arr
                pbar.update()
    return out


def read_multichannel_images(fns):
    assert 0 < len(fns)

    metadata = read(fns[0])
    n_channels = metadata['n_channels']
    img_shape = read(metadata['filenames'][0]).shape

    data = np.zeros((len(fns), *img_shape[:-1], metadata['n_channels']), dtype=np.uint8)
    for idx, fn in enumerate(fns):
        metadata = read(fn)
        imgs = read_arrays(metadata['filenames'])
        for img_idx, img in enumerate(imgs):
            ch = 3
            if n_channels < (img_idx+1)*3:
                ch = n_channels - img_idx*3
            data[idx, ..., img_idx * 3:min((img_idx + 1) * 3, n_channels)] = img[..., :ch]
    return data


###########################
# Functions to save files #
###########################

def save_json(out_fn, data):
    with open(out_fn, 'w') as f:
        json.dump(data, f)


def save(path, data, varname=None):
    """
    Saves the variable ``var`` to the given path. The file format depends on the file extension.
    List of supported file types:

    - .npy: numpy
    - .png, .jpg, .gif: pictures
    - .txt: text file, one element per line. ``var`` must be a string or list of strings.
    """
    make_directory(os.path.dirname(path))
    if path.endswith('.json'):
        with open(path, 'w') as f:
            return json.dump(data, f)
    elif path.endswith(('.npy', '.flo')):
        np.save(path, data)
    elif path.endswith(('.png', '.jpg')):
        # cv2.imwrite(path, var)
        # source: https://jdhao.github.io/2019/09/11/opencv_unicode_image_path/
        is_success, im_buf_arr = cv2.imencode(os.path.splitext(path)[-1], np.array(data))
        im_buf_arr.tofile(path)
    elif path.endswith(('.txt', '.csv', '.xlsx')):
        if isinstance(data, np.ndarray):
            np.savetxt(path, data, delimiter=',')
        else:
            with open(path, 'w') as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    for line in data:
                        if isinstance(line, list):
                            f.write('\t'.join([str(e) for e in line]))
                        else:
                            f.write(str(line))
                        f.write('\n')
        if path.endswith('.xlsx'):
            read_file = pd.read_csv(path)
            read_file.to_excel(path, index=None, header=False)
    elif path.endswith('.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(data, f, 2)
    elif path.endswith('.mat'):
        assert varname is not None, 'when using matlab format the variable name must be defined'
        scipy.io.savemat(path, {varname: data})
    else:
        raise NotImplementedError('Unknown extension: ' + os.path.splitext(path)[1])


def save_files(out_dir, data, writer=save, extension='.png', fn_format='{:05d}{}', start_idx=0):
    make_directory(out_dir)
    for i in range(len(data)):
        writer(os.path.join(out_dir, fn_format.format(start_idx + i, extension)), data[i])


def save_images(out_dir, images, **kwargs):
    return save_files(out_dir, images, writer=save, **kwargs)


def save_multichannel_images(out_dir, images, writer=save, extension='.png', fn_format='{:05d}{}', start_idx=0):
    ch = images[0].shape[-1]

    make_directory(out_dir)
    for i in range(len(images)):
        metadata = {'n_channels': ch, 'filenames': []}
        for ch_idx in range(0, ch, 3):
            img = images[i][..., ch_idx:ch_idx+3]
            if img.shape[-1] < 3:
                tmp = np.zeros((*img.shape[:-1], 3), dtype=img.dtype)
                tmp[..., :img.shape[-1]] = img
                img = tmp
            img_fn = os.path.join(out_dir, fn_format.format(start_idx + i, 'ch' + str(ch_idx) + extension))
            writer(img_fn, img)
            metadata['filenames'].append(img_fn)
        writer(os.path.join(out_dir, fn_format.format(start_idx + i, '.json')), metadata)


def save_video(out_fn, frames, fps=10.0, shape=None):
    assert out_fn.endswith(('.avi', '.mp4'))
    make_directory(os.path.dirname(out_fn))

    if out_fn.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if shape is not None:
        h, w = shape[:2]
    else:
        h, w = frames[0].shape[:2]
    vid_out = cv2.VideoWriter(out_fn, fourcc, fps, (w, h))

    for frame in frames:
        vid_out.write(frame)
    vid_out.release()


def save_video_parallel(out_fn, read_frame, frame_params, frame_shape,
                        fps=10.0, length=None, chunksize=100, n_processes=-1):
    assert out_fn.endswith(('.avi', '.mp4'))
    make_directory(os.path.dirname(out_fn))

    if length is None:
        length = len(frame_params)
    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, max(1, length//chunksize))
    if n_processes == 1:
        return save_video(out_fn, frame_params, fps=fps, shape=frame_shape)

    if out_fn.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frame_shape[:2]
    vid_out = cv2.VideoWriter(out_fn, fourcc, fps, (w, h))
    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm(total=length) as pbar:
            for frame in pool.imap(read_frame, frame_params, chunksize=chunksize):
                vid_out.write(frame)
                pbar.update()
    vid_out.release()


def merge_videos(out_fn, video_fns):
    videos = [VideoFileClip(v) for v in video_fns]
    final_video = concatenate_videoclips(videos)
    final_video.to_videofile(out_fn, fps=30, remove_temp=False, codec="libx264")


####################################################
# Functions to manipulate the file and directories #
####################################################

def make_directory(dir_name):
    """Creates the given directory if it isn't exists.

    Parameters
    ----------
    dir_name : str
        Path to a directory
    """
    if dir_name == '':
        return
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def make_directories(*args):
    """Creates the given directories if they aren't exists."""
    for dir_name in args:
        make_directory(dir_name)


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        print("[Warning]: file {} is not a file or dir.".format(path))


def remove_files(fns):
    for fn in fns:
        remove(fn)


def copy_directory(src, dst, symlinks=False, ignore=None):
    """https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth"""
    make_directory(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def copy_file(src, dst, overwrite=True):
    if not overwrite and os.path.exists(dst):
        return
    make_directory(os.path.dirname(dst))
    shutil.copy2(src, dst)


def copy_files(src_list, dst_list, overwrite=True):
    assert len(src_list) == len(dst_list)

    for src, dst in zip(src_list, dst_list):
        copy_file(src, dst, overwrite=overwrite)


def modify(fn, func):
    save(fn, func(read(fn)))


def modify_parallel(fns, func, chunksize=100, n_processes=-1):
    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, max(1, len(fns)//chunksize))
    if n_processes == 1:
        for fn in fns:
            modify(fn, func)

    with multiprocessing.Pool(processes=n_processes) as pool:
        with tqdm(total=len(fns)) as pbar:
            for _ in enumerate(pool.imap(partial(modify, func=func), fns, chunksize=chunksize)):
                pbar.update()
