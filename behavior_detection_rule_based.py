import warnings
import numpy as np
import os
import utils.io
import utils.segmentation
from matplotlib import pyplot as plt


FPS = 30
BEHAVIOR_DT = 0.2  # Behavior annotation happens every 0.2 seconds
step_frames = int(FPS * BEHAVIOR_DT)
HEAD_ID = 1; BODY_ID = 2; TAIL_ID = 3
H_KP_ID = 0; BOT_KP_ID = 1; EOT_KP_ID = 2
SHAPE = (420, 640)  # (H, W)
DPI = 200
PAD = 10
shape_pad = (SHAPE[0] + 2 * PAD, SHAPE[1] + 2 * PAD)


ALIGN_BOT_TH, ALIGN_H_TH = 12, 15
CLOSE_TH_1, CLOSE_TH_2 = .15, .09
TOUCH_TH = .02
CHASING_TH = .5
MOUNTING_TH = 2000  # 1700
SIDE_L_TH_1, SIDE_L_TH_2, SIDE_D_TH, PASSIVE_TH_1, PASSIVE_TH_2 = .5, .75, .125, .07, .2  # .35, .5, .1, .07, .2
PARALLEL_TH = 60
HEAD_D_TH = .1  # .075
EPS = .009



def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__') and not isinstance(obj, str)


def _clip(arr, shape=SHAPE):
    ''' Clip values so that they are inside the original shape after removing the padding '''
    
    if arr.size == 0:
        return arr
    h, w = shape
    arr[:, 0], arr[:, 1] = arr[:, 0].clip(PAD, h + PAD - 1), arr[:, 1].clip(PAD, w + PAD - 1)
    return arr


def get_files(dir_):  # Using imap returns bytes instead of string
    return np.array(utils.io.list_directory(dir_, fn_constraint=lambda fn: not fn.endswith('.pkl')), dtype=object)


def get_dirs(dir_):
    return np.array(utils.io.list_directory(dir_, only_dirs=True), dtype=object)


def read_frame_instances(instances_file):
    instances_frame = utils.io.read(instances_file)
    if instances_frame.shape[-1] == 3:
        instances_frame = utils.segmentation.multi2singlechannel(instances_frame[None, ...])[0]
    instance_ids = np.unique(instances_frame)[1:]
    if len(instance_ids) != 2:
        return np.zeros((2,) + SHAPE, dtype=bool), f'!!! #instances = {len(instance_ids)}'
    instance_1 = instances_frame == instance_ids[0]
    instance_2 = instances_frame == instance_ids[1]
    return np.array([instance_1, instance_2]), ''


def to_arch(behaviors, base_arch_file, contain_markers=False):
    import re

    with open(base_arch_file, mode='r') as file:
        arch_file = file.read().lower()

    behavior_list = re.findall(pattern='cbname\t(.+)\n', string=arch_file)  # CBName
    if not contain_markers:
        rep_str = ''.join([f'{behavior_list.index(b1.lower()) + 1}\t\t{behavior_list.index(b2.lower()) + 1}\n' for (b1, b2) in behaviors])
    else:
        rep_str = ''.join([f'{behavior_list.index(b1.lower()) + 1}\t{(behavior_list.index(m.lower()) + 1) if m != "" else ""}\t{behavior_list.index(b2.lower()) + 1}\n' for (b1, m, b2) in behaviors])
    return re.sub(pattern='(?s)(0.000\t).*', repl=f'\g<1>{rep_str}', string=arch_file)


def from_arch(arch_file, contain_markers=False):
    import re

    with open(arch_file, mode='r') as file:
        arch_file = file.read().lower()

    behavior_list = re.findall(pattern='cbname\t(.+)\n', string=arch_file)  # CBName
    behavior_list.append('')
    behaviors = re.search(pattern='(?s)0.000\t(.*)', string=arch_file).group(1)
    if contain_markers:
        id_list = np.array(re.split('\t|\n',  behaviors), dtype='<U50')
        id_list[id_list == ''] = f"{len(behavior_list)}"
        return np.array(behavior_list, dtype='<U50')[id_list[:-1].astype(int).reshape(-1, 3) - 1]
    else:
        return np.array(behavior_list, dtype='<U50')[np.array(behaviors.split(), dtype=int).reshape(-1, 2) - 1]


def merge_arch(arch_dir):
    files = get_files(arch_dir)
    arch_files = list(filter(lambda x: os.path.splitext(os.path.basename(x))[1] == '.arch', files))
    behaviors = np.vstack([from_arch(arch_file=arch_file) for arch_file in arch_files])
    behaviors = np.vstack([behaviors, behaviors[-1]])  # Needed in Solomon Coder, maybe due to error in video generation!!!
    return to_arch(behaviors=behaviors, base_arch_file=arch_files[0])


def set_ax(ax, shape=SHAPE, scale=None, title=None):
    STEP = 50
    h, w = shape
    ax.set_aspect(aspect=1.)
    ax.set_xlim(0, w)
    ax.xaxis.set_ticks(np.arange(0, w, STEP))
    ax.set_ylim(h, 0)
    ax.yaxis.set_ticks(np.arange(0, h, STEP)[::-1])
    
    if title is not None:
        ax.set_title(title)
    
    if scale is not None:
        # Set axis size as a scale factor of shape for the given dpi (by setting figsize)
        h_, w_ = [scale * dim / ax.figure.dpi for dim in [h, w]]
        l, b, r, t = ax.get_position().get_points().flatten()
        figw = w_ / (r - l)
        figh = h_ / (t - b)
        ax.figure.set_size_inches((figw, figh))


def to_mask(points, shape=None):
    points = np.array(points).round().astype(int)
    if shape is None:
        i_max, j_max = points.max(axis=0)
        shape = (i_max + 2, j_max + 2)
    mask = np.zeros(shape, dtype='bool')
    mask[points[:, 0], points[:, 1]] = 1
    return mask


def find_boundary(mask):
    from skimage.measure import find_contours
    
    contours = find_contours(mask, level=.5)
    return np.vstack(contours) if contours else np.array([])


def curve_length(points):
    from skimage.morphology import skeletonize

    points = np.array(points)
    if points.size == 0:
        return .0
    return skeletonize(to_mask(points)).sum()


def find_ends(mask):
    from scipy.ndimage import generic_filter
    from skimage.morphology import skeletonize

    def end_filter(kernel):
        return (kernel[4] == 1 and kernel.sum() == 2 or\
                kernel[4] == 1 and kernel.sum() == 3 and kernel[[0, 2, 6, 8]].sum() == 1 and\
                1 in np.diff(np.where(kernel)[0]) and np.isin(np.diff(np.where(kernel)[0]), [2, 3]).sum() == 1)

    return np.argwhere(generic_filter(skeletonize(mask), end_filter, (3, 3)))


def find_medoid(points, metric='euclidean'):
    from scipy.spatial.distance import pdist, squareform
    from skimage.morphology import skeletonize
    from skimage.graph import route_through_array

    points = np.array(points)
    if metric == 'curve_length':
        weights = 3 - 2 * skeletonize(to_mask(points))
        metric = lambda u, v: route_through_array(weights, u, v)[1]
    return points[squareform(pdist(points, metric=metric)).sum(axis=0).argmin()]


def find_midpoint(points):
    from skimage.morphology import skeletonize
    from skimage.graph import route_through_array

    points = np.array(points)
    mask = to_mask(points)
    ends = find_ends(mask)
    weights = 3 - 2 * skeletonize(mask)
    if len(ends) != 2:
        print(f'!!! find_midpoint() error, len(ends) != 2 ({len(ends)}), using find_medoid() instead')
        return find_medoid(points, metric='curve_length')
    cost_fun = lambda x: route_through_array(weights, ends[0], x)[1]
    costs = np.array(list(map(cost_fun, points)))
    points = points[costs > (cost_fun(ends[1]) / 2)]
    costs = costs[costs > (cost_fun(ends[1]) / 2)]
    return points[costs.argsort()[0]]


def find_angle(v1, v2):
    if np.all(v1 == 0) or np.all(v2 == 0):
        return np.nan
    v1, v2 = np.array([v1, v2])
    with np.errstate(divide='ignore'):
        return np.rad2deg(np.arctan(np.cross(v1, v2) / v1.dot(v2)))


def find_touching(img_labeled):
    from scipy.ndimage import generic_filter

    def touching_filter(kernel):
        return kernel[4] == 1 and np.any(kernel == 2)

    return np.argwhere(generic_filter(img_labeled, function=touching_filter, size=(3, 3)))


def find_closest_coordinate(coordinates, closest_to, furthest_from=None, closest_to_points=None): 
    from scipy.spatial.distance import cdist
    
    assert coordinates.ndim == 2
    assert coordinates.shape[1] == 2
    assert len(closest_to) == 2
    assert furthest_from is None or len(furthest_from) == 2
    assert closest_to_points is None or closest_to_points.shape[-1] == 2

    distances = np.linalg.norm(coordinates - np.array(closest_to).reshape(1, 2), ord=2, axis=1)
    if furthest_from is not None:
        distances -= np.linalg.norm(coordinates - np.array(furthest_from).reshape(1, 2), ord=2, axis=1)
    if closest_to_points is not None:
        min_to_points = np.min(cdist(coordinates, closest_to_points), axis=1)
        distances += min_to_points
    return coordinates[np.argmin(distances), :]


def find_furthest_coordinate(coordinates, furthest_from, weights, closest_to=None):
    from skimage.graph import route_through_array
    from numpy.linalg import norm
    
    distances = np.array(list(map(lambda x: route_through_array(weights, furthest_from, x)[1], coordinates)))
    if closest_to is not None:
        distances -= norm(coordinates - closest_to, axis=1)
    return coordinates[distances.argmax()]


def draw_annotations(ax, instances, kps, rat_a=1, kp_ids=[H_KP_ID, BOT_KP_ID, EOT_KP_ID]):
    rat_b = 1 if rat_a == 2 else 2
    instance_a, instance_b = instances[rat_a - 1], instances[rat_b - 1]
    #kps = kps[:, kp_ids]
    kps_a, kps_b = kps[rat_a - 1], kps[rat_b - 1]
    
    # Visualization
    s_l = .2; c_a = 'g'; c_h_a = 'lime'; c_b_a = 'b'; c_b = 'y'; c_h_b = 'gold'; c_b_b = 'b'; c_t = 'r'; c_t_a = 'r'
    c_t_b = 'r'; alpha = .5

    ax.scatter(*np.where(instance_a)[::-1], c=c_a, s=s_l, alpha=alpha, label=f'rat_{rat_a}', zorder=0)
    ax.scatter(*np.where(instance_b)[::-1], c=c_b, s=s_l, alpha=alpha, label=f'rat_{rat_b}', zorder=0)

    for kp_id in kp_ids:
        s_a = s_b = 7; lw_a = lw_b = .5
        if kp_id == H_KP_ID:
            c_a_ = c_h_a; c_b_ = c_h_b
        elif kp_id == BOT_KP_ID:
            c_a_ = c_b_a; c_b_ = c_b_b
        elif kp_id == EOT_KP_ID:
            c_a_ = c_t_a; c_b_ = c_t_b
        ax.scatter(kps_a[kp_id, -1], kps_a[kp_id, -2], c=c_a_, s=s_a, edgecolors='black', lw=lw_a, zorder=2)
        ax.scatter(kps_b[kp_id, -1], kps_b[kp_id, -2], c=c_b_, s=s_b, edgecolors='black', lw=lw_b, zorder=2)
        ax.legend(scatterpoints=10)


def find_kps(seq_dir, output_dir, kps_init=None, verbose=0):
    def remove_tail(instances, kps_prev=None, verbose=0):
        from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects, medial_axis, skeletonize
        from skimage.measure import label, regionprops
        from numpy.linalg import norm
        from matplotlib import pyplot as plt
        
        def find_tail_score(region_img, instance):
            _, dist_map = medial_axis(instance, return_distance=True)
            skel = skeletonize(instance)
            skel = skel & region_img
            dist_on_skel = dist_map * skel
            if dist_on_skel[dist_on_skel > 0].size == 0:
                return 0
            percentile = np.percentile(dist_on_skel[dist_on_skel > 0], q=75)
            skel = np.where(dist_on_skel <= percentile, skel, 0)
            return skel.sum() / percentile

        SMALL_TH = 30
        OPENING_TH = 8
        CLOSING_TH = 8
        TAIL_TH = 1.5
        TAIL_CERTAIN_TH = 5
        
        instances = remove_small_objects(instances.astype(bool), min_size=SMALL_TH)
        instances_after = np.zeros_like(instances)
        instances_stage_1, tails_stage_1 = np.zeros_like(instances), np.zeros_like(instances)
        for i, instance in enumerate(instances):
            instance_opened = binary_opening(instance, selem=disk(OPENING_TH))
            tail = (instance & ~instance_opened)
            tail = remove_small_objects(tail.astype(bool), min_size=SMALL_TH)
            instances_stage_1[i], tails_stage_1[i] = (instance & ~tail), tail
        tails = tails_stage_1.copy()
        tails_closed = np.zeros_like(tails)
        tail_certain = np.array([False, False])
        tail_missed = np.array([False, False])
        for i in range(len(tails)):
            tails_closed[i] = binary_closing(tails[i], selem=disk(CLOSING_TH))  # If the tail is disconnected
            tail_labeled = label(tails_closed[i])
            instance_closed = binary_closing(instances[i], selem=disk(CLOSING_TH))
            regions = sorted(regionprops(tail_labeled), key=lambda x: find_tail_score(tail_labeled == x.label,
                                                                                    instance_closed), reverse=True)
            for region in regions:
                region_img = tail_labeled == region.label
                tail_score = find_tail_score(region_img, instance_closed)
                tail_certain[i] |= (tail_score > TAIL_CERTAIN_TH)
                # plt.figure(); plt.title(f'i={i}, tail_score={tail_score}');
                # plt.imshow(region_img);
                if tail_score > TAIL_TH:
                    if kps_prev is not None:
                        dist2head = norm(region.centroid - kps_prev[i, H_KP_ID])
                        dist2bot = norm(region.centroid - kps_prev[i, BOT_KP_ID])
                        tail_close2bot = dist2bot < dist2head
                    else:
                        tail_close2bot = True
                    if tail_close2bot:
                        tails[i] = region_img & instances[i]
                        tail_missed[i] = False
                        break
                    else:
                        tail_missed[i] = True
                tails[i] &= ~region_img
        instances_after = instances & ~tails
        if verbose:
            fig, axes = plt.subplots(6, 2, figsize=(9, 22))
            axes[0, 0].set_title('original - rat1'); axes[0, 0].imshow(instances[0])
            axes[0, 1].set_title('original - rat2'); axes[0, 1].imshow(instances[1])
            axes[1, 0].set_title('after_stage_1 - rat1'); axes[1, 0].imshow(instances_stage_1[0])
            axes[1, 1].set_title('after_stage_1 - rat2'); axes[1, 1].imshow(instances_stage_1[1])
            axes[2, 0].set_title('tails_stage_1 - rat1'); axes[2, 0].imshow(tails_stage_1[0])
            axes[2, 1].set_title('tails_stage_1 - rat2'); axes[2, 1].imshow(tails_stage_1[1])
            axes[3, 0].set_title('tails_closed - rat1'); axes[3, 0].imshow(tails_closed[0])
            axes[3, 1].set_title('tails_closed - rat2'); axes[3, 1].imshow(tails_closed[1])
            axes[4, 0].set_title('after - rat1'); axes[4, 0].imshow(instances_after[0])
            axes[4, 1].set_title('after - rat2'); axes[4, 1].imshow(instances_after[1])
            axes[5, 0].set_title('tails - rat1'); axes[5, 0].imshow(tails[0])
            axes[5, 1].set_title('tails - rat2'); axes[5, 1].imshow(tails[1])
            fig.tight_layout()
        return instances_after, tail_certain, tail_missed

    def find_kps_frame(instances, kps_prev=None, state_old=None, verbose=0):  
        from numpy.linalg import norm
        from skimage.morphology import medial_axis, skeletonize
        from skimage.feature import corner_peaks, corner_shi_tomasi
        from automatic_annotation import select_furthest_coordinates
        from skimage.graph import route_through_array
        from matplotlib import pyplot as plt

        D_MAX = 100
        CLOSE_R = 1.5
        correction_freq = 2
        kps = np.zeros((2, 2, 3), dtype=int)
        if state_old is None:
            state_old = {
                'i': 0,
                'kps_prev': kps_prev,
                'error_h_1': 0,
                'error_bot_1': 0,
                'error_h_2': 0,
                'error_bot_2': 0,
                'tail_correction_1': False,
                'tail_correction_2': False,
                'tail_missed_1': 0,
                'tail_missed_2': 0,
            }
        kps_prev = state_old['kps_prev']
        instances_pad = np.zeros((2, *shape_pad), dtype=bool)
        instances_pad[:, PAD:-PAD, PAD:-PAD] = instances
        if kps_prev is not None:
            kps_prev = kps_prev[..., -2:] + PAD
        else:
            state_old['tail_correction_1'] = True
            state_old['tail_correction_2'] = True
        for i in [0, 1]:
            if state_old[f'tail_correction_{i + 1}'] and state_old[f'tail_missed_{i + 1}'] >= correction_freq:
                kps_prev[i, [H_KP_ID, BOT_KP_ID]] = kps_prev[i, [BOT_KP_ID, H_KP_ID]]
                state_old[f'tail_missed_{i + 1}'] = 0
        instances_ntail, tail_certain, tail_missed = remove_tail(instances_pad.copy(), kps_prev=kps_prev)
        for i in [0, 1]:
            if tail_certain[i]:
                if tail_missed[i]:
                    state_old[f'tail_correction_{i + 1}'] = True
                else:
                    state_old[f'tail_correction_{i + 1}'] = False
            if tail_missed[i]:
                state_old[f'tail_missed_{i + 1}'] += 1
            else:
                state_old[f'tail_missed_{i + 1}'] = 0

        if verbose > 1:
            c_boundary = 'black'; c_touch = 'r'; c_skel = 'black'; c_end = 'black'; c_corner = 'cyan'; c_close = 'r'
            s_l = .2; s_1 = 2; s_2 = 5
            h, w = SHAPE
            fig, ax = plt.subplots(1, 1, figsize=(10, 10 * h / w), dpi=DPI)
            set_ax(ax)
        warnings = []
        for i, (instance, instance_ntail) in enumerate(zip(instances_pad, instances_ntail)):
            _, dist_map = medial_axis(instance, return_distance=True)
            skel = skeletonize(instance)
            skel[dist_map < np.mean(dist_map[dist_map > 0])] = 0
            skel &= instance_ntail
            ends_skel = _clip(find_ends(skel))
            ends_skel = ends_skel if len(ends_skel) >= 2 else _clip(np.argwhere(skel))
            boundary_ntail = _clip(find_boundary(instance_ntail))
            corners_ntail = _clip(corner_peaks(corner_shi_tomasi(instance_ntail), min_distance=2))
            corners_ntail = corners_ntail if len(corners_ntail) >= 2 else boundary_ntail
            img_labeled = instance.astype(int) + instance_ntail
            touch_points = _clip(find_touching(img_labeled))
            kp_bot_0 = kp_bot_1 = kp_bot = kp_head_0 = close_points = kp_head = None
            if len(ends_skel) >= 2 and len(corners_ntail) >= 2:
                if touch_points.size > 0:  # Tail is taken off
                    kp_bot = find_midpoint(touch_points)
                else:
                    ends2_skel = select_furthest_coordinates(ends_skel, weights=(3 - 2 * skel))
                    if kps_prev is not None:
                        kp_bot_0 = find_closest_coordinate(ends2_skel, closest_to=kps_prev[i, BOT_KP_ID])
                        kp_bot_1 = find_furthest_coordinate(corners_ntail, furthest_from=kps_prev[i, H_KP_ID],
                                                            weights=(2 - skel), closest_to=kp_bot_0)
                    else:
                        p1 = find_furthest_coordinate(corners_ntail, furthest_from=ends2_skel[0], weights=(2 -  skel),
                                                        closest_to=ends2_skel[1])
                        p0 = find_furthest_coordinate(corners_ntail, furthest_from=ends2_skel[1], weights=(2 - skel),
                                                        closest_to=ends2_skel[0])
                        r0 = norm(p0 - ends2_skel[0]) / dist_map[ends2_skel[0][0], ends2_skel[0][1]]
                        r1 = norm(p1 - ends2_skel[1]) / dist_map[ends2_skel[1][0], ends2_skel[1][1]]
                        if r0 > r1:
                            kp_bot_0, kp_bot_1 = ends2_skel[0], p0
                        else:
                            kp_bot_0, kp_bot_1 = ends2_skel[1], p1
                    kp_bot = np.average([kp_bot_0, kp_bot_1], axis=0,
                                        weights=(norm(kp_bot_0 - kp_bot_1), 2 * dist_map[kp_bot_0[0], kp_bot_0[1]]))
                
                kp_head_0 = find_furthest_coordinate(ends_skel, furthest_from=kp_bot, weights=(2 - 1 * skel))
                dist_head = dist_map[kp_head_0[0], kp_head_0[1]]
                close_points = boundary_ntail[norm(kp_head_0 - boundary_ntail, axis=1) < CLOSE_R * dist_head].astype(int)
                if len(close_points) >= 2:
                    mask_close = to_mask(close_points)
                    ends_close = find_ends(mask_close)
                    ends_close = ends_close if len(ends_close) >= 2 else close_points
                    ends2_close = select_furthest_coordinates(ends_close, weights=(3 - 2 * mask_close))
                    close_points_, _ = route_through_array(10 - 9 * to_mask(boundary_ntail, shape=shape_pad), ends2_close[0],
                                                        ends2_close[1])
                    close_points = np.vstack([close_points, close_points_])
                    kp_head = find_midpoint(close_points)
                else:
                    warnings.append(f'!!! len(close_points) < 2 ({len(close_points)}) for rat_{i + 1}')
            if (kp_head is None or kp_bot is None) and len(corners_ntail) >= 2:
                if len(ends_skel) < 2:
                    warnings.append(f'!!! len(ends_skel) < 2 ({len(ends_skel)}) for rat_{i + 1}')
                if kps_prev is not None:
                    kp_bot = find_furthest_coordinate(corners_ntail, furthest_from=kps_prev[i, H_KP_ID], weights=(2 - skel),
                                                      closest_to=kps_prev[i, BOT_KP_ID]) if kp_bot is None else kp_bot
                    kp_head = find_furthest_coordinate(corners_ntail, furthest_from=kps_prev[i, BOT_KP_ID], weights=(2 - skel),
                                                       closest_to=kps_prev[i, H_KP_ID]) if kp_head is None else kp_head
                elif kp_bot is not None:
                    kp_head = find_furthest_coordinate(corners_ntail, furthest_from=kp_bot, weights=(np.ones_like(skel)))
            if kp_head is None or kp_bot is None:
                warnings = [f'!!! Key-points could not be found for rat_{i + 1}',
                            f'len(ends_skel) < 2 ({len(ends_skel)}) or len(corners_ntail) < 2 ({len(corners_ntail)})']
                if verbose > 1:
                    ax.text(x=.01, y=(.01 + i * .03), s='    '.join(warnings), size=8, ha='left', transform=ax.transAxes)
                kps = None
                break
            
            if kps_prev is not None:
                if norm(kp_head - kps_prev[i, H_KP_ID]) > D_MAX and state_old[f'error_h_{i + 1}'] < correction_freq:
                    kp_head = kps_prev[i, H_KP_ID]
                    state_old[f'error_h_{i + 1}'] += 1
                else:
                    state_old[f'error_h_{i + 1}'] = 0
                if norm(kp_bot - kps_prev[i, BOT_KP_ID]) > D_MAX and state_old[f'error_bot_{i + 1}'] < correction_freq:
                    kp_bot = kps_prev[i, BOT_KP_ID]
                    state_old[f'error_bot_{i + 1}'] += 1
                else:
                    state_old[f'error_bot_{i + 1}'] = 0
            kps[i, H_KP_ID, -2:], kps[i, BOT_KP_ID, -2:] = kp_head - PAD, kp_bot - PAD

            if verbose < 2:
                continue

            # Visualization
            ii, jj = zip(*(boundary_ntail - PAD)); ax.scatter(jj, ii, c=c_boundary, s=s_l, alpha=.4)
            if touch_points.size > 0:
                ii, jj = zip(*(touch_points - PAD)); ax.scatter(jj, ii, c=c_touch, s=s_2, alpha=.5, marker='*')
            ii, jj = np.where(skel[PAD:-PAD, PAD:-PAD]); ax.scatter(jj, ii, c=c_skel, s=s_l, alpha=.4)
            if kp_head_0 is not None:
                ii, jj = kp_head_0 - PAD; ax.scatter(jj, ii, c=c_end, s=s_2, alpha=.5, marker='*')
            if kp_bot_0 is not None:
                ii, jj = kp_bot_0 - PAD; ax.scatter(jj, ii, c=c_end, s=s_2, alpha=.5, marker='*')
                ii, jj = kp_bot_1 - PAD; ax.scatter(jj, ii, c='b', s=s_2, alpha=.5, marker='*')
            if corners_ntail.size > 0:
                ii, jj = zip(*(corners_ntail - PAD)); ax.scatter(jj, ii, c=c_corner, s=s_1, alpha=0.8)
            if close_points is not None and close_points.size > 0:
                ii, jj = zip(*(close_points - PAD)); ax.scatter(jj, ii, c=c_close, s=s_1, alpha=.5, marker='*')
            ax.text(x=.01, y=(.01 + i * .03), s='    '.join(warnings), size=8, ha='left', transform=ax.transAxes)

        instances_ntail = instances_ntail[:, PAD:-PAD, PAD:-PAD]
        state_old['kps_prev'] = kps
        state_old['i'] += 1

        if verbose < 2:
            ax = None
        return instances_ntail, kps, state_old, ax, warnings
    
    # Find key-points in the input sequence frames
    seq_name = os.path.basename(seq_dir)
    instances_files = get_files(seq_dir)
    instances_improved, kps_improved = [], []
    kps_prev = state_old = None
    kps_prev = np.zeros((2, 2, 3), dtype=int)
    if kps_init is not None:
        kps_prev = np.zeros((2, 2, 3), dtype=int)
        kps_prev[..., -2:] = np.array(kps_init)
    print(f'\n{seq_name}\nFinding key-points...\n')
    for frame_id in range(0, len(instances_files), step_frames):
        try:
            frame_name = os.path.splitext(os.path.basename(instances_files[frame_id]))[0]
            title = f'sequence: {seq_name}    frame: {frame_name}'
            if verbose:
                print(title)
            instances_, err = read_frame_instances(instances_files[frame_id])
            warnings = []
            ax = None
            if not err:
                instances_, kps_, state_old, ax, warnings = find_kps_frame(instances_, kps_prev=kps_prev, state_old=state_old,
                                                                           verbose=verbose)
            if err or kps_ is None:
                kps_ = np.zeros((2, 2, 3), dtype=int)
                if kps_prev is not None:
                    kps_ = kps_prev
            else:
                kps_prev = kps_
            if err or warnings:
                print('\n'.join([err] + warnings), f'\tin\t{seq_name}\t{frame_name}')
            if verbose > 1 and ax is not None:
                draw_annotations(ax=ax, instances=instances_, kps=kps_, rat_a=1, kp_ids=[H_KP_ID, BOT_KP_ID])
                ax.set_title(title)
                dir_ = os.path.join(output_dir, 'images/kps_detection', seq_name)
                utils.io.make_directory(dir_)
                plt.savefig(f'{dir_}/{frame_name}.png')
                plt.close()
            state_old['kps_prev'] = kps_prev
        except:
            pass
        finally:
            instances_improved.append(instances_)
            kps_improved.append(kps_prev)

    return np.array(instances_improved), np.array(kps_improved)


def detect_behaviors(seq_dir, output_dir, instances_improved, kps_improved=None, verbose=0):
    def detect_behaviors_frame(instances, kps, title='', state_old=None, verbose=verbose):
        from scipy.spatial.distance import cdist
        from skimage.morphology import medial_axis, skeletonize
        from skimage.graph import route_through_array
        from numpy.linalg import norm
        from matplotlib import pyplot as plt
        from matplotlib.legend_handler import HandlerPathCollection

        def check_rat(rat_a):
            if rat_a == 1:
                head_a, head_b = head_1, head_2
                bot_a, bot_b = bot_1, bot_2
                area_a, area_b = area_1, area_2
                close_len_a, close_len_b = close_len_1, close_len_2
                kps_d_a, kps_d_b = kps_d_1, kps_d_2
                instance_ch_a, instance_ch_b = instance_ch_1, instance_ch_2
                head_close2head_b = head_close2head_2
            else:
                head_a, head_b = head_2, head_1
                bot_a, bot_b = bot_2, bot_1
                area_a, area_b = area_2, area_1
                close_len_a, close_len_b = close_len_2, close_len_1
                kps_d_a, kps_d_b = kps_d_2, kps_d_1
                instance_ch_a, instance_ch_b = instance_ch_2, instance_ch_1
                head_close2head_b = head_close2head_1

            # Check chasing behavior
            theta_bot = find_angle(head_a - bot_a, bot_b - bot_a)
            theta_head = find_angle(head_a - bot_a, head_b - bot_a)
            distances = np.hstack([cdist([bot_a], [head_a, bot_b, head_b])[0], norm(bot_b - head_a)])
            if theta_bot * theta_head > 0:
                aligned_a = abs(theta_bot) < ALIGN_BOT_TH and abs(theta_head) < ALIGN_H_TH
            else:
                aligned_a = abs(theta_bot) < ALIGN_BOT_TH and abs(theta_bot) + abs(theta_head) < ALIGN_H_TH
            aligned_a &= ((distances[0] < 1.1 * distances[1] < distances[2]) and (distances[3] < distances[1]))
            possible_chasing = not touch and aligned_a and (distances[3] / skel_mean) < CHASING_TH
            chasing = possible_chasing and kps_d_a[0] > PASSIVE_TH_1 and kps_d_b[0] > PASSIVE_TH_1

            # Find aligned_b for the Side-to-side contact detection
            theta_bot = find_angle(head_b - bot_b, bot_a - bot_b)
            theta_head = find_angle(head_b - bot_b, head_a - bot_b)
            distances = np.hstack([cdist([bot_b], [head_b, bot_a, head_a])[0], norm(bot_a - head_b)])
            if theta_bot * theta_head > 0:
                aligned_b = abs(theta_bot) < ALIGN_BOT_TH and abs(theta_head) < ALIGN_H_TH
            else:
                aligned_b = abs(theta_bot) < ALIGN_BOT_TH and abs(theta_bot) + abs(theta_head) < ALIGN_H_TH
            aligned_b &= ((distances[0] < 1.1 * distances[1] < distances[2]) and (distances[1] > distances[3]))

            close_mounting = (close_len_a > SIDE_L_TH_1 or close_len_b > SIDE_L_TH_1)
            mounted = close_mounting and area_a < MOUNTING_TH
            mounting = close_mounting and area_b < MOUNTING_TH

            passive = kps_d_a.sum() < PASSIVE_TH_1 and kps_d_a.sum() <= kps_d_b.sum() or\
                      instance_ch_a < PASSIVE_TH_2 and instance_ch_a <= instance_ch_b

            body_sniffing = np.inf > head_close2head_b > HEAD_D_TH        
            
            behavior = 'Non-social behaviour'
            if chasing:
                behavior = 'Chasing'
            if close:
                if mounted:
                    behavior = 'Being mounted'
                elif mounting:
                    behavior = 'Mounting'
                elif side_contact and not aligned_a and not aligned_b:
                    if passive:
                        behavior = 'Side-to-side contact'  # 'Passive side-to-side contact'
                    else:
                        behavior = 'Side-to-side contact'
                elif head2head:
                    behavior = 'Head-to-head contact'
                elif body_sniffing:
                    behavior = 'Body sniffing'
            if behavior == 'Non-social behaviour' and possible_chasing:
                behavior = 'possible_chasing'
            behavior += f' {rat_a}'

            state_variables = {
                'theta_bot': theta_bot,
                'theta_head': theta_head,
                'aligned': aligned_a,
                'passive': passive,
            }
            return behavior, state_variables

        # Starting point of the rule-based algorithm
        boundary_1, boundary_2 = find_boundary(instances[0]), find_boundary(instances[1])

        print('Check instances and kps:', seq_dir, instances is not None, kps is not None)
        if not instances.any() or not kps.any() or boundary_1.size == 0 or boundary_2.size == 0:
            return np.array(['Non-social behaviour 1', 'Non-social behaviour 2'], dtype='<U50'), state_old

        """
        ALIGN_BOT_TH, ALIGN_H_TH = 12, 15
        CLOSE_TH_1, CLOSE_TH_2 = .15, .09
        TOUCH_TH = .02
        CHASING_TH = .5
        MOUNTING_TH = 1400  # TODO make it smaller  # 1700
        SIDE_L_TH_1, SIDE_L_TH_2, SIDE_D_TH, PASSIVE_TH_1, PASSIVE_TH_2 = .35, .5, .1, .07, .2
        PARALLEL_TH = 60
        HEAD_D_TH = .1    # TODO make it larger  # .075
        EPS = .009
        """

        if state_old is None:
            state_old = {
                'instances': instances,
                'kps': kps,
            }
        
        head_1, bot_1 = kps[0, H_KP_ID, -2:], kps[0, BOT_KP_ID, -2:]
        head_2, bot_2 = kps[1, H_KP_ID, -2:], kps[1, BOT_KP_ID, -2:]

        # Calculate skeleton lengths
        _, dist_map = medial_axis(instances[0], return_distance=True)
        skel_1 = skeletonize(instances[0])
        skel_1[dist_map < np.mean(dist_map[dist_map > 0])] = 0
        skel_points = np.array(route_through_array(3 - 2 * skel_1, start=bot_1, end=head_1)[0])
        skel_1.fill(0)
        skel_1[skel_points[:, 0], skel_points[:, 1]] = 1
        _, skel_len_1 = route_through_array(3 - 2 * skel_1, start=bot_1, end=head_1)

        _, dist_map = medial_axis(instances[1], return_distance=True)
        skel_2 = skeletonize(instances[1])
        skel_2[dist_map < np.mean(dist_map[dist_map > 0])] = 0
        skel_points = np.array(route_through_array(3 - 2 * skel_2, start=bot_2, end=head_2)[0])
        skel_2.fill(0)
        skel_2[skel_points[:, 0], skel_points[:, 1]] = 1
        _, skel_len_2 = route_through_array(3 - 2 * skel_2, start=bot_2, end=head_2)

        skel_mean = np.mean([skel_len_1, skel_len_2])

        # Find closest contact points
        distances_1_2 = cdist(boundary_1, boundary_2)
        argmin_1, argmin_2 = np.unravel_index(distances_1_2.argmin(), distances_1_2.shape)
        min_1_2 = distances_1_2[argmin_1, argmin_2]
        closest_1, closest_2 = boundary_1[argmin_1], boundary_2[argmin_2]

        # Find all close points
        argclose_1, argclose_2 = np.where(distances_1_2 / skel_mean < CLOSE_TH_2)
        close_points_1, close_points_2 = boundary_1[argclose_1], boundary_2[argclose_2]
        close_len_1, close_len_2 = min(curve_length(close_points_1), curve_length(close_points_2)) /\
                                   np.array([skel_len_1, skel_len_2])

        # Find head-close points
        head_close_points_1 = boundary_1[np.where(norm(head_2 - boundary_1, axis=1) / skel_len_2 < CLOSE_TH_1)]
        head_close_points_2 = boundary_2[np.where(norm(head_1 - boundary_2, axis=1) / skel_len_1 < CLOSE_TH_1)]

        # Find all touching points
        argtouch_1, argtouch_2 = np.where(distances_1_2 / skel_mean < TOUCH_TH)
        touch_points_1, touch_points_2 = boundary_1[argtouch_1], boundary_2[argtouch_2]
        touch_len_1, touch_len_2 = min(curve_length(touch_points_1), curve_length(touch_points_2)) /\
                                   np.array([skel_len_1, skel_len_2])

        # Calculate closeness
        distance = min_1_2 / skel_mean
        close = distance < CLOSE_TH_1
        touch = distance < TOUCH_TH

        area_1, area_2 = instances[0].sum(), instances[1].sum()

        # Calculate closeness to H and BoT key-points
        if close_points_1.size > 0 and close_points_2.size > 0:
            close2head_1 = np.min(norm(close_points_1 - head_1, axis=1)) / skel_len_1
            close2head_2 = np.min(norm(close_points_2 - head_2, axis=1)) / skel_len_2
            close2bot_1 = np.min(norm(close_points_1 - bot_1, axis=1)) / skel_len_1
            close2bot_2 = np.min(norm(close_points_2 - bot_2, axis=1)) / skel_len_2
        else:
            close2head_1 = norm(closest_1 - head_1) / skel_len_1
            close2head_2 = norm(closest_2 - head_2) / skel_len_2
            close2bot_1 = norm(closest_1 - bot_1) / skel_len_1
            close2bot_2 = norm(closest_2 - bot_2) / skel_len_2

        if head_close_points_1.size > 0:
            head_close2head_1 = np.min(norm(head_close_points_1 - head_1, axis=1)) / skel_len_1
            head_close2bot_1 = np.min(norm(head_close_points_1 - bot_1, axis=1)) / skel_len_1
        else:
            head_close2head_1 = head_close2bot_1 = np.inf
        if head_close_points_2.size > 0:
            head_close2head_2 = np.min(norm(head_close_points_2 - head_2, axis=1)) / skel_len_2
            head_close2bot_2 = np.min(norm(head_close_points_2 - bot_2, axis=1)) / skel_len_2
        else:
            head_close2head_2 = head_close2bot_2 = np.inf

        parallel_angle = find_angle(head_1 - bot_1, head_2 - bot_2)
        parallel = abs(parallel_angle) < PARALLEL_TH

        kps_d_1, kps_d_2 = norm((kps - state_old['kps'])[..., -2:], axis=-1) / skel_mean
        instance_ch_1, instance_ch_2 = np.sum(np.isin(state_old['instances'] + (2 * instances), [1, 2]),
                                              axis=(1, 2)) / np.mean([area_1, area_2])

        # Check behaviors
        side_contact = parallel and ((close_len_1 > SIDE_L_TH_1 and close_len_2 > SIDE_L_TH_1 and\
                                      (close2head_1 > SIDE_D_TH or close2head_2 > SIDE_D_TH) and\
                                      (close2bot_1 > SIDE_D_TH or close2bot_2 > SIDE_D_TH)) or\
                                     (close_len_1 > SIDE_L_TH_2 and close_len_2 > SIDE_L_TH_2))

        head2head = (head_close2head_1 < HEAD_D_TH and head_close2head_2 < np.inf or\
                     head_close2head_2 < HEAD_D_TH and head_close2head_1 < np.inf)

        # Detect behaviors
        behavior_1, state_variables = check_rat(rat_a=1)
        theta_bot_1, theta_head_1, aligned_1, passive_1 = list(state_variables.values())
        behavior_2, state_variables = check_rat(rat_a=2)
        theta_bot_2, theta_head_2, aligned_2, passive_2 = list(state_variables.values())

        state_old = {
            'kps': kps,
            'instances': instances,
            'behaviors': np.array([behavior_1, behavior_2], dtype='<U50'),
        }
        if verbose < 2:
            return state_old['behaviors'], state_old

        # Visualization
        h, w = SHAPE
        fig = plt.figure(figsize=(9, 9 * h / w), dpi=DPI)
        ax = fig.add_axes([.05, .15, .95, .75])
        title += f'\ndt: {state_old["behaviors"]}'
        set_ax(ax=ax, title=title)
        if verbose > 2:
            fig.set_size_inches((9, 9 * h / w + 3))
            ax.set_position([.05, .25, .95, .75])
            ax.set_title(title, fontsize=9)
            text = f'\ntheta_bot_1={theta_bot_1:.1f},\ttheta_bot_2={theta_bot_2:.1f},\t'\
                   f'theta_head_1={theta_head_1:.1f},\ttheta_head_2={theta_head_2:.1f},\t'\
                   f'aligned_1={aligned_1},\taligned_2={aligned_2}\n'\
                   f'ALIGN_BOT_TH={ALIGN_BOT_TH},\tALIGN_H_TH={ALIGN_H_TH},\n'\
                   f'kps_d_1={kps_d_1.round(3)},\tkps_d_2={kps_d_2.round(3)},\t'\
                   f'instance_ch_1={instance_ch_1.round(3)},\tinstance_ch_2={instance_ch_2.round(3)},\t'\
                   f'passive_1={passive_1},\tpassive_2={passive_2},\tPASSIVE_TH_1={PASSIVE_TH_1},\t'\
                   f'PASSIVE_TH_2={PASSIVE_TH_2},\tCHASING_TH={CHASING_TH}\n'\
                   f'\n'\
                   f'distance={distance:.3f},\tclose={close},\tCLOSE_TH_1={CLOSE_TH_1},\tCLOSE_TH_2={CLOSE_TH_2}\n'\
                   f'\n'\
                   f'area_1={area_1},\tarea_2={area_2},\tMOUNTING_TH={MOUNTING_TH}\n'\
                   f'\n'\
                   f'close2head_1={close2head_1:.3f},\tclose2head_2={close2head_2:.3f},\t'\
                   f'close2bot_1={close2bot_1:.3f},\tclose2bot_2={close2bot_2:.3f}\n'\
                   f'head_close2head_1={head_close2head_1:.3f},\thead_close2head_2={head_close2head_2:.3f},\t'\
                   f'head_close2bot_1={head_close2bot_1:.3f},\thead_close2bot_2={head_close2bot_2:.3f},\t'\
                   f'HEAD_D_TH={HEAD_D_TH}\n'\
                   f'parallel_angle={parallel_angle:.2f},\tparallel={parallel},\t'\
                   f'close_len_1={close_len_1:.3f},\tclose_len_2={close_len_2:.3f},\tside_contact={side_contact}\n'\
                   f'SIDE_D_TH={SIDE_D_TH},\tSIDE_L_TH_1={SIDE_L_TH_1},\tSIDE_L_TH_2={SIDE_L_TH_2}\n'\
                   f'\n'\
                   f'touch_len_1={touch_len_1:.3f},\ttouch_len_2={touch_len_2:.3f},\t'\
                   .replace('\t', '   ')
            ax.text(x=0, y=-.45, s=text, size=8, ha='left', transform=ax.transAxes)

        alpha = .8; s = .2; c_skel = 'black'; alpha_skel = .4; c_close = 'orange'; c_touch = 'r'; c_head_close = 'b'
        c_closest='black'

        draw_annotations(ax=ax, instances=instances, kps=kps, rat_a=1, kp_ids=[H_KP_ID, BOT_KP_ID])

        ax.scatter(*np.where(skel_1)[::-1], c=c_skel, alpha=alpha_skel, s=s, label='skel')
        ax.scatter(*np.where(skel_2)[::-1], c=c_skel, alpha=alpha_skel, s=s)

        ax.scatter(close_points_1[:, 1], close_points_1[:, 0], c=c_close, s=s, label='close')
        ax.scatter(close_points_2[:, 1], close_points_2[:, 0], c=c_close, s=s)

        ax.scatter(head_close_points_1[:, 1], head_close_points_1[:, 0], c=c_head_close, s=s, label='head_close')
        ax.scatter(head_close_points_2[:, 1], head_close_points_2[:, 0], c=c_head_close, s=s)

        ax.scatter(touch_points_1[:, 1], touch_points_1[:, 0], c=c_touch, s=s, label='touch')
        ax.scatter(touch_points_2[:, 1], touch_points_2[:, 0], c=c_touch, s=s)

        s_ = ax.scatter(closest_1[1], closest_1[0], marker='+', s=15, lw=.5, c=c_closest, alpha=alpha, label='closest')
        ax.scatter(closest_2[1], closest_2[0], marker='+', s=15, lw=.5, c=c_closest, alpha=alpha)

        ax.legend(scatterpoints=10, handler_map={s_: HandlerPathCollection(numpoints=1)})
        return state_old['behaviors'], state_old

    # Detect social behaviors in the input sequence frames
    seq_name = os.path.basename(seq_dir)
    instances_files = get_files(seq_dir)
    results = []
    state_old = None
    print(f'\n{seq_name}\nDetecting social behaviors...\n')
    if verbose:
        print(f'Frame{"":35}Predicted behaviors\n{"-" * 120}')
    for frame_id in range(0, len(instances_files), step_frames):
        frame_name = os.path.splitext(os.path.basename(instances_files[frame_id]))[0]
        title = f'sequence: {seq_name}    frame: {frame_name}'
        instances_ = instances_improved[frame_id // step_frames]
        kps_= kps_improved[frame_id // step_frames]
        behaviors_pred, state_old = detect_behaviors_frame(instances_, kps_, title, state_old=state_old, verbose=verbose)
        if verbose:
            print(f'{title}\t{behaviors_pred}')
            if verbose > 1:
                dir_ = os.path.join(output_dir, 'images/behavior_detection', seq_name)
                utils.io.make_directory(dir_)
                plt.savefig(f'{dir_}/{frame_name}.png')
                plt.close()
        results.append(behaviors_pred)

    # Fix chasing
    for i in range(len(results)):
        behaviors_pred = results[i]
        if 'possible_chasing 1' in behaviors_pred or 'possible_chasing 2' in behaviors_pred:
            rat_id = 1 if 'possible_chasing 1' in behaviors_pred else 2
            behaviors_pred_next = results[min(i + 1, len(results) - 1)]
            results[i][rat_id - 1] = f'Chasing {rat_id}' if f'Chasing {rat_id}' in behaviors_pred_next\
                                                         else f'Non-social behaviour {rat_id}'
    return np.array(results)


def process_wrapper(params, output_dir, base_arch_file, verbose):
    seq_id, seq_dir, kps_init = params
    instances_improved, kps_improved = find_kps(seq_dir=seq_dir, output_dir=output_dir, kps_init=kps_init, verbose=verbose)
    # instances_improved = np.load(f'{output_dir}/instances_improved.npy', allow_pickle=True)[seq_id]
    # kps_improved = np.load(f'{output_dir}/kps_improved.npy', allow_pickle=True)[seq_id]
    behaviors_dt = detect_behaviors(seq_dir=seq_dir, output_dir=output_dir, instances_improved=instances_improved,
                                    kps_improved=kps_improved, verbose=verbose)
    dir_ = os.path.join(output_dir, 'behaviors_dt')
    utils.io.make_directory(dir_)
    with open(os.path.join(dir_, f'{os.path.basename(seq_dir)}.arch'), mode='w') as file:
        file.write(to_arch(behaviors=behaviors_dt, base_arch_file=base_arch_file))
    return seq_id, instances_improved, kps_improved


def evaluate_behaviors(behaviors_gt_dir, behaviors_dt_dir, output_dir, instances_files=None, labels=None):
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    from matplotlib import pyplot as plt

    def remove_rat_index(arr):
        def is_integer(n):
            try:
                float(n)
            except ValueError:
                return False
            else:
                return float(n).is_integer()

        shape = arr.shape
        return np.array(list(map(lambda behavior: behavior[:-2] if is_integer(behavior[-1]) else behavior,
                                 arr.flatten())), dtype='<U50').reshape(shape)

    files = get_files(behaviors_gt_dir)
    arch_files_gt = list(filter(lambda x: os.path.splitext(os.path.basename(x))[1] == '.arch', files))
    
    files = get_files(behaviors_dt_dir)
    arch_files_dt = list(filter(lambda x: os.path.splitext(os.path.basename(x))[1] == '.arch', files))
    
    behaviors_gt, behaviors_dt, errors = [], [], []
    for seq_id, (arch_file_gt, arch_file_dt) in enumerate(zip(arch_files_gt, arch_files_dt)):
        seq_name = os.path.splitext(os.path.basename(arch_file_gt))[0]
        # print(seq_name)
        for frame_id, (behaviors_gt_, behaviors_dt_) in enumerate(zip(from_arch(arch_file_gt), from_arch(arch_file_dt))):
            behaviors_gt.append(behaviors_gt_)
            behaviors_dt.append(behaviors_dt_)
            if np.any(behaviors_gt_ != behaviors_dt_):
                if instances_files is not None:
                    frame_name = os.path.splitext(os.path.basename(instances_files[seq_id][frame_id * step_frames]))[0]
                else:
                    frame_name = f'{frame_id * step_frames}'
                errors.append([seq_name, frame_name])
            else:
                errors.append(['NA', 'NA'])
    # print(behaviors_gt[:10], behaviors_dt[:10])
    behaviors_gt, behaviors_dt, errors = np.array(behaviors_gt), np.array(behaviors_dt), np.array(errors)
    behaviors_gt, behaviors_dt = remove_rat_index(behaviors_gt), remove_rat_index(behaviors_dt)
    # print(behaviors_gt[:10], behaviors_dt[:10])

    # Evaluate detections
    if labels is None:
        labels = np.unique(behaviors_gt)

    # print(behaviors_gt.shape, labels, np.unique(behaviors_dt))
    mask = np.isin(behaviors_gt, labels).all(axis=1)
    behaviors_gt, behaviors_dt, errors = behaviors_gt[mask], behaviors_dt[mask], errors[mask]
    errors = errors[np.all(errors != 'NA', axis=1)]
    # print(behaviors_gt.shape, behaviors_gt[:10], behaviors_dt[:10])
    # behaviors_gt = behaviors_gt[:, ::-1]

    dt_str = f'time-step of {round(BEHAVIOR_DT, 3)} sec'
    print(f'\nEvaluation on {dt_str}...\n')
    confusion_mat = confusion_matrix(
        y_true=behaviors_gt.flatten(),
        y_pred=behaviors_dt.flatten(),
        labels=labels,
    )
    report = classification_report(
        y_true=behaviors_gt.flatten(),
        y_pred=behaviors_dt.flatten(),
        labels=labels,  # only for displaying the report not for evaluation
        zero_division=0,
    )
    
    # Display results
    print(report)
    with open(os.path.join(output_dir, 'classification_report.txt'), mode='w') as file:
        file.write(report)
    
    fig = plt.figure(figsize=(7, 7), dpi=DPI)
    ax = fig.add_axes([.3, .1, .7, .9])
    ax.set_title(dt_str)
    ConfusionMatrixDisplay(confusion_mat, display_labels=labels).plot(ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    with open(os.path.join(output_dir, 'errors.txt'), mode='w') as file:
        file.write(f'{errors}')
    return


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--base_arch_file', required=True, help='Base coding sheet directory.')
    parser.add_argument('--eval', default=False, action='store_true', help='Evaluate previous results or detect then evaluate.')
    parser.add_argument('-bgt', '--behaviors_gt_dir', help='Ground-truth behaviors directory.')
    parser.add_argument('-bdt', '--behaviors_dt_dir', help='Detected behaviors directory.')
    parser.add_argument('-i', '--instances_dir', help='Instances directory.')
    parser.add_argument('-o', '--output_dir', help='Output directory.')
    parser.add_argument('-ki', '--kps_init', help='List of key-points of the initial frames in the input sequences.')
    parser.add_argument('--merge', action='store_true', help='Merge the output .arch files.')
    parser.add_argument('-v', '--verbose', default=1, help='1 >> show progress, 2 >> visualize process, 3 >> more details.')
    return parser.parse_args()


def run(base_arch_file, eval=False, behaviors_gt_dir=None, behaviors_dt_dir=None, instances_dir=None, output_dir=None,
        kps_init=None, merge=False, verbose=1, n_processes=-1):
    '''instances_dir = '/home/ahmad/rats_behavior/behavior_detection_test/exp1/instances'
    base_arch_file = '/home/ahmad/rats_behavior/behavior_detection_test/base.arch'
    kps_init = f"[None, None, None, None, [[[380, 399], [330, 373]], [[278, 385], [360, 400]]], None, None, None, None, None,"\
               f" None, None, [[[330, 373], [380, 399]], [[400, 330], [330, 327]]], None, None, None, None, None]"'''
    if not eval:
        assert instances_dir is not None and os.path.isdir(instances_dir), 'Please provide a correct instances_dir'
        if output_dir is None:
            output_dir = os.path.join(instances_dir, os.pardir, 'output')
    else:
        assert behaviors_gt_dir is not None and os.path.isdir(behaviors_gt_dir), 'Please provide a correct behaviors_gt_dir'
        assert output_dir is not None and os.path.isdir(output_dir), 'Please provide a correct output_dir'
    labels = [
        'Chasing', 'Head-to-head contact', 'Body sniffing', 'Side-to-side contact', 'Passive side-to-side contact',
        'Mounting', 'Being mounted', 'Non-social behaviour',
    ]
    instances_files = None if instances_dir is None else [[instances_file for instances_file in get_files(seq_dir)]\
                                                                          for seq_dir in get_dirs(instances_dir)]
    if eval:
        if behaviors_dt_dir is not None:
            assert os.path.isdir(behaviors_dt_dir), 'Please provide a correct behaviors_dt_dir'
            evaluate_behaviors(behaviors_gt_dir=behaviors_gt_dir, behaviors_dt_dir=behaviors_dt_dir, output_dir=output_dir,
                               instances_files=instances_files, labels=labels)
            return
    seq_dirs = get_dirs(instances_dir)
    if kps_init is None:
        kps_init = [None] * len(seq_dirs)
    else:
        import ast
        kps_init = ast.literal_eval(kps_init)
    verbose = int(verbose)

    import multiprocessing, functools
    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, len(seq_dirs))

    with multiprocessing.Pool(processes=n_processes) as pool:
        instances_improved, kps_improved = [0] * len(seq_dirs), [0] * len(seq_dirs)
        for seq_id, instances_improved_, kps_improved_\
        in pool.imap_unordered(functools.partial(process_wrapper, output_dir=output_dir, base_arch_file=base_arch_file,
                                                 verbose=verbose),
                               zip(range(len(seq_dirs)), seq_dirs, kps_init)):
            instances_improved[seq_id] = instances_improved_
            kps_improved[seq_id] = kps_improved_
        # np.save(f'{output_dir}/instances_improved.npy', np.array(instances_improved, dtype=object))
        # np.save(f'{output_dir}/kps_improved.npy', np.array(kps_improved, dtype=object))
    
    if merge:
        with open(os.path.join(output_dir, 'behaviors_dt.arch'), mode='w') as file:
            file.write(merge_arch(arch_dir=os.path.join(output_dir, 'behaviors_dt')))

    if behaviors_gt_dir is not None:
        behaviors_dt_dir = os.path.join(output_dir, 'behaviors_dt')
        evaluate_behaviors(behaviors_gt_dir=behaviors_gt_dir, behaviors_dt_dir=behaviors_dt_dir, output_dir=output_dir,
                           instances_files=instances_files, labels=labels)


if __name__ == "__main__":
    args = parse_args()
    run(**vars(args))
