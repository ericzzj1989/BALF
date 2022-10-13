
import cv2
import numpy as np
import torch
from torchgeometry.core import warp_perspective

def create_common_region_masks(h_dst_2_src, shape_src, shape_dst):
    
    # Create mask. Only take into account pixels in the two images
    inv_h = np.linalg.inv(h_dst_2_src)
    inv_h = inv_h / inv_h[2, 2]

    # Applies mask to destination. Where there is no 1, we can no find a point in source.
    ones_dst = np.ones((shape_dst[0], shape_dst[1]))
    ones_dst = remove_borders(ones_dst, borders=15)
    mask_src = cv2.warpPerspective(ones_dst, h_dst_2_src, (shape_src[1], shape_src[0]))
    mask_src = np.where(mask_src >= 0.75, 1.0, 0.0)
    mask_src = remove_borders(mask_src, borders=15)

    ones_src = np.ones((shape_src[0], shape_src[1]))
    ones_src = remove_borders(ones_src, borders=15)
    mask_dst = cv2.warpPerspective(ones_src, inv_h, (shape_dst[1], shape_dst[0]))
    mask_dst = np.where(mask_dst >= 0.75, 1.0, 0.0)
    mask_dst = remove_borders(mask_dst, borders=15)

    return mask_src, mask_dst

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

def apply_homography_to_points(points, h):
    new_points = []

    for point in points:

        new_point = h.dot([point[0], point[1], 1.0])

        tmp = point[2]**2+np.finfo(np.float32).eps

        Mi1 = [[1/tmp, 0], [0, 1/tmp]]
        Mi1_inv = np.linalg.inv(Mi1)
        Aff = getAff(point[0], point[1], h)

        BMB = np.linalg.inv(np.dot(Aff, np.dot(Mi1_inv, np.matrix.transpose(Aff))))

        [e, _] = np.linalg.eig(BMB)
        new_radious = 1/((e[0] * e[1])**0.5)**0.5

        new_point = [new_point[0] / new_point[2], new_point[1] / new_point[2], new_radious, point[3]]
        new_points.append(new_point)

    return np.asarray(new_points)

def getAff(x,y,H):
    h11 = H[0,0]
    h12 = H[0,1]
    h13 = H[0,2]
    h21 = H[1,0]
    h22 = H[1,1]
    h23 = H[1,2]
    h31 = H[2,0]
    h32 = H[2,1]
    h33 = H[2,2]
    fxdx = h11 / (h31 * x + h32 * y + h33) - (h11 * x + h12 * y + h13) * h31 / (h31 * x + h32 * y + h33) ** 2
    fxdy = h12 / (h31 * x + h32 * y + h33) - (h11 * x + h12 * y + h13) * h32 / (h31 * x + h32 * y + h33) ** 2

    fydx = h21 / (h31 * x + h32 * y + h33) - (h21 * x + h22 * y + h23) * h31 / (h31 * x + h32 * y + h33) ** 2
    fydy = h22 / (h31 * x + h32 * y + h33) - (h21 * x + h22 * y + h23) * h32 / (h31 * x + h32 * y + h33) ** 2

    Aff = [[fxdx, fxdy], [fydx, fydy]]

    return np.asarray(Aff)

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



def create_common_region_masks_tensor(h_dst_2_src, shape_src, shape_dst):
    
    # Create mask. Only take into account pixels in the two images
    inv_h = torch.inverse(h_dst_2_src)
    inv_h = inv_h / inv_h[2, 2]

    # Applies mask to destination. Where there is no 1, we can no find a point in source.
    ones_dst = torch.ones((h_dst_2_src.shape[0], 1, shape_dst[0], shape_dst[1])).to(h_dst_2_src.device)
    dst_shape = ones_dst.shape[2:]
    ones_dst = remove_borders_tensor(ones_dst, borders=15)
    mask_src = warp_perspective(ones_dst, h_dst_2_src, dst_shape)
    mask_src = torch.where(mask_src >= 0.75, 1.0, 0.0)
    mask_src = remove_borders_tensor(mask_src, borders=15)

    ones_src = torch.ones((h_dst_2_src.shape[0], 1, shape_src[0], shape_src[1])).to(h_dst_2_src.device)
    src_shape = ones_src.shape[2:]
    ones_src = remove_borders_tensor(ones_src, borders=15)
    mask_dst = warp_perspective(ones_src, inv_h, src_shape)
    mask_dst = torch.where(mask_dst >= 0.75, 1.0, 0.0)
    mask_dst = remove_borders_tensor(mask_dst, borders=15)

    return mask_src, mask_dst

def remove_borders_tensor(image, borders): 
    ## Input : [B, H, W, C] or [H, W, C] or [H, W]

    shape = image.shape
    new_im = torch.zeros_like(image)
    # if len(shape) == 4:
    #     shape = [shape[1], shape[2], shape[3]]
    #     new_im[:, borders:shape[0]-borders, borders:shape[1]-borders, :] = image[:, borders:shape[0]-borders, borders:shape[1]-borders, :]
    # elif len(shape) == 3:
    if len(shape) == 3:
        new_im[borders:shape[0] - borders, borders:shape[1] - borders, :] = image[borders:shape[0] - borders, borders:shape[1] - borders, :]
    else:
        new_im[borders:shape[0] - borders, borders:shape[1] - borders] = image[borders:shape[0] - borders,  borders:shape[1] - borders]
    return new_im

def get_point_coordinates_tensor(map, scale_value=1., num_points=1000, threshold=-1, order_coord='xysr'):
    ## input numpy array score map : [H, W]
    indexes = find_index_higher_scores_tensor(map, num_points=num_points, threshold=threshold)
    new_indexes = []
    for ind in range(indexes.shape[1]):

        scores = map[indexes[0,ind], indexes[1,ind]]
        if order_coord == 'xysr':
            tmp = [indexes[1,ind], indexes[0,ind], scale_value, scores]
        elif order_coord == 'yxsr':
            tmp = [indexes[0,ind], indexes[1,ind], scale_value, scores]

        new_indexes.append(tmp)

    indexes = torch.tensor(new_indexes)

    return indexes

def find_index_higher_scores_tensor(map, num_points = 1000, threshold = -1):
    # Best n points
    if threshold == -1:

        flatten = map.flatten()
        order_array, _ = torch.sort(flatten)

        order_array = torch.flip(order_array, dims=[0])

        threshold = order_array[num_points-1]
        if threshold <= 0.0:
            indexes = torch.where(order_array > 0.0)[0]
            
            if len(indexes) == 0:
                threshold = 0.0
            else:
                threshold = order_array[indexes[len(indexes)-1]]
        # elif threshold == 0.0:
        #     threshold = order_array[np.nonzero(order_array)].min()

    indexes = torch.where(map >= threshold)
    y,x = torch.where(map >= threshold)
    indexes = torch.stack([y[:num_points],x[:num_points]],dim=0)

    return indexes


def apply_homography_to_points_tensor(points, mat):
    new_points = []

    for id in range(points.shape[0]):
        denominator = points[id,0] * mat[2, 0] + points[id,1] * mat[2, 1] + mat[2, 2]
        new_x = (points[id,0] * mat[0, 0] + points[id,1] * mat[0, 1] + mat[0, 2]) / denominator
        new_y = (points[id,0] * mat[1, 0] + points[id,1] * mat[1, 1] + mat[1, 2]) / denominator

        new_point = [new_x, new_y, 1., points[id,3]]
        new_points.append(new_point)

    return torch.tensor(new_points)