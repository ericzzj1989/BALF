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
        output_score_map[pts_nms[1, idx].astype(np.int), pts_nms[0, idx].astype(np.int)] = pts_nms[2, idx]

    return output_score_map


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
