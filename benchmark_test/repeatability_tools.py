import numpy as np
from scipy.ndimage.filters import maximum_filter

import torch
import torch.nn as nn

def check_common_points(kpts, mask):
    idx_valid_points = []
    for idx, ktp in enumerate(kpts):
        if mask[int(round(ktp[0]))-1, int(round(ktp[1]))-1]:
            idx_valid_points.append(idx)
    return np.asarray(idx_valid_points)

def select_top_k(kpts, k=1000):
    scores = -1 * kpts[:, 3]
    return np.argsort(scores)[:k]

def apply_nms(score_map, size):
    
    score_map = score_map * (score_map == maximum_filter(score_map, footprint=np.ones((size, size))))

    return score_map

def apply_nms_fast(score_map, dist_thresh):
    H, W = score_map.shape
    in_corners = np.zeros((3, H*W))  # Populate point data sized 3xN.
    for idx in range(H*W):
        y_pos = idx // W
        x_pos = idx % W
        in_corners[0, idx] = x_pos
        in_corners[1, idx] = y_pos
        in_corners[2, idx] = score_map[y_pos, x_pos]

    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    # print('corners: ', corners.shape)
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
    pts_nms = out
    output_score_map = np.zeros_like(score_map)
    for idx in range(out.shape[1]):
        if pts_nms[2, idx] < 0.015:
            continue
        output_score_map[pts_nms[1, idx].astype(np.int), pts_nms[0, idx].astype(np.int)] = pts_nms[2, idx]

    return output_score_map






def get_points_direct_from_score_map(
    heatmap, conf_thresh=0.015, nms_size=15,
    subpixel=True, patch_size=5, scale_value=1., order_coord='xysr'
):

    H, W = heatmap.shape[0], heatmap.shape[1]
    ys, xs = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    if len(xs) == 0:
        return np.zeros((3, 0))
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
    print("heatmap: ", heatmap.shape)
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


'''def apply_nms_fast_tesnor(score_map, dist_thresh):
    H, W = score_map.shape
    in_corners = torch.zeros((3, H*W))  # Populate point data sized 3xN.
    for idx in range(H*W):
        y_pos = idx // W
        x_pos = idx % W
        in_corners[0, idx] = x_pos
        in_corners[1, idx] = y_pos
        in_corners[2, idx] = score_map[y_pos, x_pos]

    grid = torch.zeros((H, W)).type(torch.long) # Track NMS data.
    inds = torch.zeros((H, W)).type(torch.long)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = torch.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    # print('corners: ', corners.shape)
    rcorners = corners[:2, :].round().type(torch.long)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return torch.zeros((3, 0)).type(torch.long), torch.zeros(0).type(torch.long)
    if rcorners.shape[1] == 1:
        out = torch.stack((rcorners, in_corners[2]), dim=0).view(3,1)
        return out, torch.zeros((1)).type(torch.long)
    # Initialize the grid.
    for i in range(rcorners.shape[1]):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    m = nn.ConstantPad2d(pad, 0)
    grid = m(grid)
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i in range(rcorners.shape[1]):
        # Account for top and left padding.
        pt = (rcorners[0, i] + pad, rcorners[1, i] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = torch.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = torch.argsort(-values)
    out = out[:, inds2]
    pts_nms = out
    output_score_map = torch.zeros_like(score_map)
    for idx in range(out.shape[1]):
        if pts_nms[2, idx] < 0.015:
            continue
        output_score_map[pts_nms[1, idx].type(torch.long), pts_nms[0, idx].type(torch.long)] = pts_nms[2, idx]

    return output_score_map'''




def compute_repeatability(src_indexes, dst_indexes, overlap_err=0.4, eps=1e-6, dist_match_thresh=3, radious_size=30.):
    
    error_overlap_s = 0.
    error_overlap_m = 0.
    found_points_s = 0
    found_points_m = 0
    possible_matches = 0
    correspondences = []
    correspondences_m = []

    dst_indexes_num = len(dst_indexes)
    src_indexes_num = len(src_indexes)

    matrix_overlaps = np.zeros((len(src_indexes), len(dst_indexes)))
    matrix_overlaps_single_scale = np.zeros((len(src_indexes), len(dst_indexes)))

    max_distance = 4 * radious_size

    for idx_ref, point_ref in enumerate(src_indexes):

        radious_ref = point_ref[2]
        found_possible_match = False

        for idx_dst, point_dst in enumerate(dst_indexes):

            radious_dst = point_dst[2]
            distance = (((point_ref[0] - point_dst[0]) ** 2) + ((point_ref[1] - point_dst[1]) ** 2)) ** 0.5

            if distance <= dist_match_thresh and not found_possible_match:
                found_possible_match = True
                possible_matches += 1

            if distance > max_distance:
                continue

            factor_scale = radious_size / (max(radious_ref, radious_dst) + np.finfo(float).eps)
            I = intersection_area(factor_scale*radious_ref, factor_scale*radious_dst, distance)
            U = union_area(factor_scale*radious_ref, factor_scale*radious_dst, I) + eps

            matrix_overlaps[idx_ref, idx_dst] = I/U

            I = intersection_area(radious_size, radious_size, distance)
            U = union_area(radious_size, radious_size, I) + eps

            matrix_overlaps_single_scale[idx_ref, idx_dst] = I/U

    y_visited = np.zeros(src_indexes.shape[0], dtype=np.uint8)
    x_visited = np.zeros(dst_indexes.shape[0], dtype=np.uint8)

    # Multiply matrix to get descendent order
    for index in (-1 * matrix_overlaps_single_scale).flatten().argsort():
        y_pos = index // dst_indexes.shape[0]
        x_pos = index % dst_indexes.shape[0]
        if x_visited[x_pos] or y_visited[y_pos]:
            continue
        max_overlap = matrix_overlaps_single_scale[y_pos, x_pos]
        if max_overlap < (1 - overlap_err):
            break
        found_points_s += 1
        error_overlap_s += (1 - max_overlap)
        correspondences.append([x_pos, y_pos])
        # update visited cells
        x_visited[x_pos] = 1
        y_visited[y_pos] = 1

    matrix_overlaps_single_scale = 0
    del matrix_overlaps_single_scale

    y_visited = np.zeros(src_indexes.shape[0], dtype=np.uint8)
    x_visited = np.zeros(dst_indexes.shape[0], dtype=np.uint8)

    # Multiply matrix to get descendent order
    for index in (-1 * matrix_overlaps).flatten().argsort():
        y_pos = index // dst_indexes.shape[0]
        x_pos = index % dst_indexes.shape[0]
        if x_visited[x_pos] or y_visited[y_pos]:
            continue
        max_overlap = matrix_overlaps[y_pos, x_pos]
        if max_overlap < (1 - overlap_err):
            break
        found_points_m += 1
        error_overlap_m += (1 - max_overlap)
        correspondences_m.append([x_pos, y_pos])
        # update visited cells
        x_visited[x_pos] = 1
        y_visited[y_pos] = 1

    matrix_overlaps = 0
    del matrix_overlaps

    points = dst_indexes_num
    if src_indexes_num < points:
        points = src_indexes_num

    rep_s = (found_points_s / np.asarray(points, float)) * 100.0
    rep_m = (found_points_m / np.asarray(points, float)) * 100.0

    if found_points_m == 0:
        error_overlap_m = 0.0
    else:
        error_overlap_m = error_overlap_m / float(found_points_m+np.finfo(float).eps)

    if found_points_s == 0:
        error_overlap_s = 0.0
    else:
        error_overlap_s = error_overlap_s / float(found_points_s+np.finfo(float).eps)

    return {'rep_single_scale': rep_s, 'rep_multi_scale': rep_m, 'num_points_single_scale': found_points_s,
            'num_points_multi_scale': found_points_m, 'error_overlap_single_scale': error_overlap_s,
            'error_overlap_multi_scale': error_overlap_m, 'total_num_points': points,
            'correspondences': np.asarray(correspondences), 'possible_matches': possible_matches,
            'correspondences_m': np.asarray(correspondences_m)}

def intersection_area(R, r, d = 0):
    """Return the area of intersection of two circles.

    The circles have radii R and r, and their centres are separated by d.

    """
    if d <= abs(R-r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r)**2
    if d >= r + R:
        # The circles don't overlap at all.
        return 0

    r2, R2, d2 = r**2, R**2, d**2
    alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    beta = np.arccos((d2 + R2 - r2) / (2*d*R))

    return ( r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta)))

def union_area(r, R, intersection):
    return (np.pi * (r ** 2)) + (np.pi * (R ** 2)) - intersection
