import numpy as np
from scipy.ndimage.filters import maximum_filter

import torch
import torch.nn as nn
import torchvision

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

def apply_nms_fast(score_map, dist_thresh, conf_thresh=0.015):
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
        if pts_nms[2, idx] < conf_thresh:
            continue
        output_score_map[pts_nms[1, idx].astype(np.int), pts_nms[0, idx].astype(np.int)] = pts_nms[2, idx]

    return output_score_map


def get_nms_score_map_from_score_map(
    heatmap, conf_thresh=0.015, nms_size=15
):
    H, W = heatmap.shape[0], heatmap.shape[1]
    ys, xs = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    if len(xs) == 0:
        return np.zeros_like(heatmap)
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = xs
    pts[1, :] = ys
    pts[2, :] = heatmap[ys, xs]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_size)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.

    nms_heatmap = np.zeros_like(heatmap)
    nms_heatmap[pts[1,:],pts[0,:]] = pts[2,:]

    return nms_heatmap # H,W



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



def box_nms(prob, size=4, iou=0.1, min_prob=0.015, keep_top_k=-1):
    """
    :param prob: probability, torch.tensor, must be [1,H,W]
    :param size: box size for 2d nms
    :param iou:
    :param min_prob:
    :param keep_top_k:
    :return:
    """
    assert(prob.shape[0]==1 and len(prob.shape)==3)
    prob = prob.squeeze(dim=0)

    pts = torch.stack(torch.where(prob>=min_prob)).t()
    boxes = torch.cat((pts-size/2.0, pts+size/2.0),dim=1).to(torch.float32)
    scores = prob[pts[:,0],pts[:,1]]
    print('********boxes shape', boxes.shape)
    print('********scores shape', scores.shape)
    indices = torchvision.ops.nms(boxes=boxes.cuda(), scores=scores.cuda(), iou_threshold=iou)
    # indices = nms(boxes=boxes.cuda(), scores=scores.cuda(), iou_threshold=iou)
    pts = pts[indices,:]
    scores = scores[indices]
    if keep_top_k>0:
        k = min(scores.shape[0], keep_top_k)
        scores, indices = torch.topk(scores,k)
        pts = pts[indices,:]
    nms_prob = torch.zeros_like(prob)
    nms_prob[pts[:,0],pts[:,1]] = scores

    return nms_prob.unsqueeze(dim=0)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
 
 
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)  # 每个框的面积 (N,)
    area2 = box_area(boxes2)  # (M,)
 
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2] # N中一个和M个比较； 所以由N，M 个
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
 
    wh = (rb - lt).clamp(min=0)  # [N,M,2]  #小于0的为0  clamp 钳；夹钳；
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]  
 
    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # NxM， boxes1中每个框和boxes2中每个框的IoU值；
 
 
def nms(boxes, scores, iou_threshold):
    """
    :param boxes: [N, 4]， 此处传进来的框，是经过筛选（NMS之前选取过得分TopK）之后， 在传入之前处理好的；
    :param scores: [N]
    :param iou_threshold: 0.7
    :return:
    """
    keep = []  # 最终保留的结果， 在boxes中对应的索引；
    idxs = scores.argsort()  # 值从小到大的 索引
    while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
        # 得分最大框对应的索引, 以及对应的坐标
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        keep.append(max_score_index)
        if idxs.size(0) == 1:  # 就剩余一个框了；
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]
 
    keep = idxs.new(keep)  # Tensor
    return keep


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



def compute_resize_repeatability(
    keypoints, warped_keypoints, h, shape_src, shape_dst,
    keep_k_points=1000, distance_thresh=5):
    """
    Compute the repeatability. The experiment must contain in its output the prediction
    on 2 images, an original image and a warped version of it, plus the homography
    linking the 2 images.
    """
    def warp_keypoints(keypoints, H):
        num_points = keypoints.shape[0]
        homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))],
                                            axis=1)
        warped_points = np.dot(homogeneous_points, np.transpose(H))
        return warped_points[:, :2] / warped_points[:, 2:]

    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are
        inside the dimensions of shape. """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
               (points[:, 1] >= 0) & (points[:, 1] < shape[1])
        return points[mask, :]

    def keep_true_keypoints(points, H, shape):
        """ Keep only the points whose warped coordinates by H
        are still inside shape. """
        warped_points = warp_keypoints(points[:, [1, 0]], H)
        warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
        mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
               (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
        return points[mask, :]

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    localization_err = -1
    repeatability = 0
    H = h

    warped_keypoints = keep_true_keypoints(warped_keypoints, np.linalg.inv(H),
                                            shape_src) # dst points in common region

    # Warp the original keypoints with the true homography
    true_warped_keypoints = keypoints
    true_warped_keypoints[:,:2] = warp_keypoints(keypoints[:, [1, 0]], H)
    true_warped_keypoints = np.stack([true_warped_keypoints[:, 1],
                                      true_warped_keypoints[:, 0],
                                      true_warped_keypoints[:, 2]], axis=-1)
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape_dst) # src points in common region

    # Keep only the keep_k_points best predictions
    warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

    # Compute the repeatability
    N1 = true_warped_keypoints.shape[0] # src points number in common region
    N2 = warped_keypoints.shape[0] # dst points number in common region

    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)
    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints,
                            ord=None, axis=2)
    count1 = 0
    count2 = 0
    localization_err1 = None
    localization_err2 = None
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        count1 = np.sum(min1 <= distance_thresh) # src points found winth pixel threshold

        localization_err1 = min1[min1 <= distance_thresh]
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh) # dst points found winth pixel threshold

        localization_err2 = min2[min2 <= distance_thresh]

    if N1 + N2 > 0:
        repeatability = (count1 + count2) / (N1 + N2) * 100.0

    if count1 + count2 > 0:
        localization_err = 0
        if localization_err1 is not None:
            localization_err += (localization_err1.sum())/ (count1 + count2)
        if localization_err2 is not None:
            localization_err += (localization_err2.sum())/ (count1 + count2)
    else:
        repeatability = 0.


    return {
        'repeatability': repeatability, 'localization_err': localization_err,
        'common_src_num': N1, 'common_dst_num': N2,
        'rep_src_num': count1, 'rep_dst_num': count2
    }