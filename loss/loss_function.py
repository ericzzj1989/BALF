import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgeometry.core import warp_perspective

from utils import train_utils, common_utils


def anchor_loss(device, input, target, loss_type='softmax'):
    if loss_type == "l2":
        loss_func = nn.MSELoss(reduction="mean")
        loss = loss_func(input, target)
    elif loss_type == "softmax":
        B, C, H, W = input.shape
        criterion = nn.BCELoss(reduction='none').to(device)
        loss = criterion(nn.functional.softmax(input, dim=1), target)
        loss = loss.sum() / (B*H*W + 1e-10)
    return loss


class ScoreLoss(object):
    def __init__(self, devie, loss_config):
        self.device = devie
        self.downsample = loss_config['downsample']
        self.image_shape = loss_config['image_shape']
        self.score_shape = [1, self.image_shape[0] // self.downsample, self.image_shape[1] // self.downsample]
        self.correspond = loss_config['correspond']
        self.usp_weight = loss_config['usp_weight']
        self.position_weight = loss_config['position_weight']
        self.score_weight = loss_config['score_weight']
        self.rep_weight = loss_config['rep_weight']
        self.uni_weight = loss_config['uni_weight']
        self.eps = 1e-12

        # create mesh grid
        x = torch.arange(self.image_shape[1] // self.downsample, requires_grad=False, device=self.device)
        y = torch.arange(self.image_shape[0] // self.downsample, requires_grad=False, device=self.device)
        y, x = torch.meshgrid([y, x])
        self.cell = torch.stack([x, y], dim=0) # shape 2,256/8,256/8=2,32,32

    def loss(self, src_outputs_score_batch, dst_outputs_score_batch, src_outputs_pos_batch, dst_outputs_pos_batch, h_src_2_dst_batch, h_dst_2_src_batch, cell_size):
        batch_size = src_outputs_score_batch.shape[0]
        loss = 0
        loss_batch_array = np.zeros((2,))

        src_score_maps_batch = F.relu(train_utils.depth_to_space_without_softmax(src_outputs_score_batch, cell_size))
        dst_score_maps_batch = F.relu(train_utils.depth_to_space_without_softmax(dst_outputs_score_batch, cell_size))

        for i in range(batch_size):
            loss_batch, loss_item = self.scoreloss(
                src_score_maps_batch[i], src_outputs_pos_batch[i],
                dst_score_maps_batch[i], dst_outputs_pos_batch[i],
                h_src_2_dst_batch[i])
            loss += loss_batch
            loss_batch_array += loss_item
        
        return loss / batch_size, loss_batch_array / batch_size


    def scoreloss(self, a_score_map, a_p, b_score_map, b_p, mat):
        a_s, b_s = self.get_feature_map_score_from_score_map(a_score_map, a_p, b_score_map, b_p)

        position_a = self.get_position(a_p, self.cell, self.downsample, flag='A', mat=mat)  # c h w, where c==2
        position_b = self.get_position(b_p, self.cell, self.downsample, flag='B', mat=None)

        distance_matrix = self.get_distance_matrix(position_a, position_b)  # c h w -> c p p

        batch_loss = 0
        loss_item = []

        if self.usp_weight > 0:
            usp_loss = self.usp_weight * self.usploss(a_s, b_s, distance_matrix)
            batch_loss += usp_loss
            loss_item.append(usp_loss.item())
        else:
            loss_item.append(0.)
        
        if self.uni_weight > 0:
            uni_loss = self.uni_weight * self.uni_loss(a_p, b_p)
            batch_loss += uni_loss
            loss_item.append(uni_loss.item())
        else:
            loss_item.append(0.)

        return batch_loss, np.array(loss_item)


    def get_feature_map_score_from_score_map(self, a_score_map, a_p, b_score_map, b_p):
        a_s = torch.tensor(self.score_shape, dtype=torch.float32).to(self.device)
        b_s = torch.tensor(self.score_shape, dtype=torch.float32).to(self.device)

        a_pixel_coor = (self.cell + a_p) * self.downsample
        b_pixel_coor = (self.cell + b_p) * self.downsample

        a_pixel_coor[torch.where(a_pixel_coor>(self.image_shape[0]-0.5))] = self.image_shape[0]-1.0
        b_pixel_coor[torch.where(b_pixel_coor>(self.image_shape[1]-0.5))] = self.image_shape[1]-1.0
        
        # if (len(torch.where(a_pixel_coor.round().long()>255)[0])>0):
        #     print('a pixel_pos error 255: ', a_pixel_coor[torch.where(a_pixel_coor.round().long()>255)])
        # if (len(torch.where(b_pixel_coor.round().long()>255)[0])>0):
        #     print('b pixel_pos error 255: ', b_pixel_coor[torch.where(b_pixel_coor.round().long()>255)])

        # assert(len(torch.where(a_pixel_coor.round().long()>255)[0])==0)
        # assert(len(torch.where(b_pixel_coor.round().long()>255)[0])==0)

        a_s = a_score_map[:,a_pixel_coor[1,:,:].round().long(), a_pixel_coor[0,:,:].round().long()]
        b_s = b_score_map[:,b_pixel_coor[1,:,:].round().long(), b_pixel_coor[0,:,:].round().long()]

        assert(a_s.shape[1] == b_s.shape[1] and a_s.shape[2] == b_s.shape[2])
        assert(a_s.shape[1] == (self.image_shape[0] // self.downsample) and a_s.shape[2] == (self.image_shape[1] // self.downsample))

        return a_s, b_s

    def get_position(self, p_map, cell, downsample, flag=None, mat=None):
        res = (cell + p_map) * downsample

        if flag == 'A':
            # https://www.geek-share.com/detail/2778133699.html  提供了src->dst的计算模式
            r = torch.zeros_like(res)
            denominator = res[0, :, :] * mat[2, 0] + res[1, :, :] * mat[2, 1] + mat[2, 2]
            r[0, :, :] = (res[0, :, :] * mat[0, 0] + res[1, :, :] * mat[0, 1] + mat[0, 2]) / denominator
            r[1, :, :] = (res[0, :, :] * mat[1, 0] + res[1, :, :] * mat[1, 1] + mat[1, 2]) / denominator
            return r
        else:
            return res

    def get_distance_matrix(self, p_a, p_b):
        c = p_a.shape[0]
        reshape_pa = p_a.reshape((c, -1)).permute(1, 0)  # c h w -> c p -> p c <=> 2,32,32 -> 2,32*32 -> 32*32,2
        reshape_pb = p_b.reshape((c, -1)).permute(1, 0)

        x = torch.unsqueeze(reshape_pa[:, 0], 1) - torch.unsqueeze(reshape_pb[:, 0], 0)  # p c -> p 1 c - 1 p c -> c p p <=> 2,32*32,32*32
        y = torch.unsqueeze(reshape_pa[:, 1], 1) - torch.unsqueeze(reshape_pb[:, 1], 0)
        dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + self.eps)
        return dis # [32*32,32*32]

    def usploss(self, a_s, b_s, distance_matrix):
        reshape_as_k, reshape_bs_k, d_k = self.get_point_pair(a_s, b_s, distance_matrix)  # p -> k

        if len(d_k) != 0:
            position_k_loss = torch.mean(d_k)

            score_k_loss = torch.mean(torch.pow(reshape_as_k - reshape_bs_k, 2))

            sk_ = (reshape_as_k + reshape_bs_k) / 2
            d_ = torch.mean(d_k)
            usp_k_loss = torch.mean(sk_ * (d_k - d_))

            position_k_loss = position_k_loss * self.position_weight
            score_k_loss = score_k_loss * self.score_weight
            usp_k_loss = usp_k_loss * self.rep_weight

            total_usp = position_k_loss + score_k_loss + usp_k_loss
            return total_usp
        else:
            return torch.tensor([0]).float().to(self.device)

    def get_point_pair(self, a_s, b_s, distance_matrix):
        a2b_min_id = torch.argmin(distance_matrix, dim=1) # min sort index of b
        len_p = len(a2b_min_id) # 1024=32*32

        assert(len_p == (a_s.shape[1]*a_s.shape[2]))
        assert(len(torch.where(a2b_min_id>(len_p-1))[0]) == 0)

        id = distance_matrix[list(range(len_p)), a2b_min_id] < self.correspond # select index flag of distance < correspond
        reshape_as = a_s.reshape(-1)
        reshape_bs = b_s.reshape(-1)

        return reshape_as[id], reshape_bs[a2b_min_id[id]], distance_matrix[id, a2b_min_id[id]]


    def uni_loss(self, a_p, b_p):
        c = a_p.shape[0]
        reshape_pa = a_p.reshape((c, -1)).permute(1, 0)  # c h w -> c p -> p c where c=2
        reshape_pb = b_p.reshape((c, -1)).permute(1, 0)

        loss = (self.get_uni_xy(reshape_pa[:, 0]) + self.get_uni_xy(reshape_pa[:, 1]))
        loss += (self.get_uni_xy(reshape_pb[:, 0]) + self.get_uni_xy(reshape_pb[:, 1]))

        return loss

    def get_uni_xy(self, position):
        idx = torch.argsort(position)  # 返回的索引是0开始的 上面的方式loss会略大0.000x级别
        idx = idx.float()
        p = position.shape[0]
        uni_l2 = torch.mean(torch.pow(position - (idx / p), 2))

        return uni_l2