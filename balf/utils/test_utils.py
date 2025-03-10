import numpy as np
import torch
import yaml
from scipy.ndimage.filters import maximum_filter


def get_cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)

    return config

def make_shape_even(image):
    height, width = image.shape[0], image.shape[1]
    padh = 1 if height % 2 != 0 else 0
    padw = 1 if width % 2 != 0 else 0
    image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode='constant', constant_values=0)
    return image

def mod_padding_symmetric(image, factor=64):
    height, width = image.shape[0], image.shape[1]
    height_pad, width_pad = ((height + factor) // factor) * factor, (
        (width + factor) // factor) * factor
    padh = height_pad - height if height % factor != 0 else 0
    padw = width_pad - width if width % factor != 0 else 0
    image = np.pad(
        image, ((padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)),
        mode='constant', constant_values=0)
    return image

def remove_borders(image, borders): 
    ## Input : [B, H, W, C] or [H, W, C] or [H, W]

    shape = image.shape
    new_im = np.zeros_like(image)
    # if len(shape) == 4:
    #     shape = [shape[1], shape[2], shape[3]]
    #     new_im[:, borders:shape[0]-borders, borders:shape[1]-borders, :] = image[:, borders:shape[0]-borders, borders:shape[1]-borders, :]
    # elif len(shape) == 3:
    if len(shape) == 3:
        new_im[borders:shape[0] - borders, borders:shape[1] - borders, :] = image[borders:shape[0] - borders, borders:shape[1] - borders, :]
    else:
        new_im[borders:shape[0] - borders, borders:shape[1] - borders] = image[borders:shape[0] - borders,  borders:shape[1] - borders]
    return new_im


def apply_nms(score_map, size):
    
    score_map = score_map * (score_map == maximum_filter(score_map, footprint=np.ones((size, size))))

    return score_map

def get_point_coordinates(map, scale_value=1., num_points=1000, threshold=-1, order_coord='xysr'):
    ## input numpy array score map : [H, W]
    indexes = find_index_higher_scores(map, num_points=num_points, threshold=threshold)
    new_indexes = []
    for ind in indexes:

        scores = map[ind[0], ind[1]]
        if order_coord == 'xysr':
            tmp = [ind[1], ind[0], scale_value, scores]
        elif order_coord == 'yxsr':
            tmp = [ind[0], ind[1], scale_value, scores]

        new_indexes.append(tmp)

    indexes = np.asarray(new_indexes)

    return np.asarray(indexes)

def find_index_higher_scores(map, num_points = 1000, threshold = -1):
    # Best n points
    if threshold == -1:

        flatten = map.flatten()
        order_array = np.sort(flatten)

        order_array = np.flip(order_array, axis=0)

        threshold = order_array[num_points-1]
        if threshold <= 0.0:
            indexes = np.argwhere(order_array > 0.0)
            if len(indexes) == 0:
                threshold = 0.0
            else:
                threshold = order_array[indexes[len(indexes)-1]]
        # elif threshold == 0.0:
        #     threshold = order_array[np.nonzero(order_array)].min()

    indexes = np.argwhere(map >= threshold)

    return indexes[:num_points]

def get_points_direct_from_score_map(
    heatmap, conf_thresh=0.015, nms_size=15,
    subpixel=True, patch_size=5, scale_value=1., order_coord='xysr'
):

    H, W = heatmap.shape[0], heatmap.shape[1]
    ys, xs = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    if len(xs) == 0:
        return np.zeros((0, 4))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = xs
    pts[1, :] = ys
    pts[2, :] = heatmap[ys, xs]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_size)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.

    if subpixel:
        pts = soft_argmax_points(pts, heatmap, patch_size=patch_size)

    new_indexes = []
    for idx in range(pts.shape[1]):
        if order_coord == 'xysr':
            tmp = [pts[0,idx], pts[1,idx], scale_value, pts[2,idx]]
        elif order_coord == 'yxsr':
            tmp = [pts[1,idx], pts[0,idx], scale_value, pts[2,idx]]

        new_indexes.append(tmp)

    indexes = np.asarray(new_indexes)

    return np.asarray(indexes) # N,4

def nms_fast(in_corners, H, W, dist_thresh):
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

def soft_argmax_points(pts, heatmap, patch_size=5):
    pts = pts.transpose().copy()
    patches = extract_patch_from_points(heatmap, pts, patch_size=patch_size)
    patches = np.stack(patches)
    patches_torch = torch.tensor(patches, dtype=torch.float32).unsqueeze(0)
    patches_torch = norm_patches(patches_torch)
    patches_torch = do_log(patches_torch)
    dxdy = soft_argmax_2d(patches_torch, normalized_coordinates=False)
    points = pts
    points[:,:2] = points[:,:2] + dxdy.numpy().squeeze() - patch_size//2
    patches = patches_torch.numpy().squeeze()
    pts_subpixel = points.transpose().copy()
    return pts_subpixel.copy()

def extract_patch_from_points(heatmap, points, patch_size=5):
    if type(heatmap) is torch.Tensor:
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = heatmap.squeeze()  # [H, W]
    pad_size = int(patch_size/2)
    heatmap = np.pad(heatmap, pad_size, 'constant')
    patches = []
    ext = lambda img, pnt, wid: img[pnt[1]:pnt[1]+wid, pnt[0]:pnt[0]+wid]
    for i in range(points.shape[0]):
        patch = ext(heatmap, points[i,:].astype(int), patch_size)
        patches.append(patch)

    return patches

def soft_argmax_2d(patches, normalized_coordinates=True):
    import torchgeometry as tgm
    m = tgm.contrib.SpatialSoftArgmax2d(normalized_coordinates=normalized_coordinates)
    coords = m(patches)  # 1x4x2
    return coords

def norm_patches(patches):
    patch_size = patches.shape[-1]
    patches = patches.view(-1, 1, patch_size*patch_size)
    d = torch.sum(patches, dim=-1).unsqueeze(-1) + 1e-6
    patches = patches/d
    patches = patches.view(-1, 1, patch_size, patch_size)
    return patches

def do_log(patches):
    patches[patches<0] = 1e-6
    patches_log = torch.log(patches)
    return patches_log