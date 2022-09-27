import os
import logging
import time
import numpy as np
import math
import glob
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from benchmark_test import repeatability_tools, geometry_tools
from loss import loss_function
from utils import common_utils


def build_optimizer(params, optim_cfg):
    logging.info("=> Setting %s solver", optim_cfg['name'])

    if optim_cfg['name'] == 'adam':
        optimizer = optim.Adam(params, lr=optim_cfg['lr'], weight_decay=optim_cfg['weight_decay'])
    elif optim_cfg['name'] == 'sgd':
        optimizer = optim.SGD(params, lr=optim_cfg['lr'])
    else:
        raise ValueError("Optimizer [%s] not recognized." % optim_cfg['name'])
    return optimizer


# def build_scheduler(optimizer, scheduler_cfg):
#     if scheduler_cfg['name'] == 'plateau':
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                             mode='min',
#                                                             patience=scheduler_cfg['patience'],
#                                                             factor=scheduler_cfg['factor'],
#                                                             min_lr=scheduler_cfg['min_lr'])
#     elif scheduler_cfg['name'] == 'sgdr':
#         scheduler = WarmRestart(optimizer)
#     elif scheduler_cfg['name'] == 'linear':
#         scheduler = LinearDecay(optimizer,
#                                 min_lr=scheduler_cfg['min_lr'],
#                                 max_epoch=scheduler_cfg['max_epoch'],
#                                 start_epoch=scheduler_cfg['start_epoch'])
#     else:
#         raise ValueError("Scheduler [%s] not recognized." % scheduler_cfg['name'])
#     return scheduler

class WarmRestart(optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, T_max=30, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 for
                base_lr in self.base_lrs]


# class LinearDecay(optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, max_epoch, start_epoch=0, min_lr=0, last_epoch=-1):
#         self.max_epoch = max_epoch
#         self.start_epoch = start_epoch
#         self.min_lr = min_lr
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         if self.last_epoch < self.start_epoch:
#             return self.base_lrs
#         return [base_lr - ((base_lr - self.min_lr) / self.max_epoch) * (self.last_epoch - self.start_epoch) for
#                 base_lr in self.base_lrs]


def build_scheduler(optimizer, total_epochs, last_epoch, optim_cfg):
    decay_epochs = [x * total_epochs for x in optim_cfg['decay_step_list']]
    warmip_epoch = optim_cfg['warmup_epoch']

    def lr_lbmd(cur_epoch):
        cur_decay = 1
        if cur_epoch < warmip_epoch:
            # cur_decay = (cur_epoch + 1) * (cur_epoch + 1) / (warmip_epoch * warmip_epoch)
            cur_decay = (cur_epoch + 1) / warmip_epoch

        for decay_epoch in decay_epochs:
            if cur_epoch >= decay_epoch:
                cur_decay = cur_decay * optim_cfg['lr_decay']

        return max(cur_decay, optim_cfg['lr_clip'] / optim_cfg['lr'])

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

    return lr_scheduler



class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

def depth_to_space(score_maps, cell_size):
    is_batch = False
    if len(score_maps.shape) == 4:
        is_batch = True

    if is_batch:
        maps = nn.functional.softmax(score_maps, dim=1) # [batch, 65, Hc, Wc]
        nodust = maps[:, :-1, :, :]
    else:
        maps = nn.functional.softmax(score_maps, dim=0) # [65, Hc, Wc]
        nodust = maps[:-1, :, :].unsqueeze(0)

    depth2space = DepthToSpace(cell_size)
    scores = depth2space(nodust)
    scores = scores.squeeze(0) if not is_batch else scores

    return scores # [1, H, W] or # [batch, 1, H, W]

def depth_to_space_without_softmax(score_maps, cell_size=8):
    is_batch = False
    if len(score_maps.shape) == 4:
        is_batch = True

    if is_batch:
        nodust = score_maps[:, :-1, :, :]
    else:
        nodust = score_maps[:-1, :, :].unsqueeze(0)

    depth2space = DepthToSpace(cell_size)
    scores = depth2space(nodust)
    scores = scores.squeeze(0) if not is_batch else scores

    return scores # [1, H, W] or # [batch, 1, H, W]

def space_to_depth(heatmaps, cell_size=8, add_dustbin=True):
    is_batch = True
    if len(heatmaps.shape) == 3:
        is_batch = False
        heatmaps = heatmaps.unsqueeze(0)

    batch_size, _, H, W = heatmaps.shape # labels [batch, 1, H, W]
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(cell_size)
    heatmaps = space2depth(heatmaps)
    if add_dustbin:
        dustbin = heatmaps.sum(dim=1)
        dustbin = 1 - dustbin
        dustbin[dustbin < 1.] = 0
        heatmaps = torch.cat((heatmaps, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        dn = heatmaps.sum(dim=1)
        score_maps = heatmaps.div(torch.unsqueeze(dn, 1))
    
    score_maps = score_maps.squeeze(0) if not is_batch else score_maps
    return score_maps # [batch, 65, Hc, Wc]

def check_val_repeatability(dataloader, device, model, cell_size=8, nms_size=15, num_points=25):
    rep_s = []
    rep_m = []
    error_overlap_s = []
    error_overlap_m = []
    possible_matches = []

    model.eval()

    iterate = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")

    ba = 0; cb =0; dc= 0; ed= 0; fe=0
    for _, batch in iterate:
        a = time.time()
        images_src_batch, images_dst_batch, heatmap_src_patch, heatmap_dst_patch, h_src_2_dst_batch, h_dst_2_src_batch = batch
        
        src_outputs = model(images_src_batch.to(device))
        dst_outputs = model(images_dst_batch.to(device))

        src_score_maps = F.relu(depth_to_space_without_softmax(src_outputs, cell_size))
        dst_score_maps = F.relu(depth_to_space_without_softmax(dst_outputs, cell_size))

        b = time.time()
        # hom = geo_tools.prepare_homography(h_dst_2_src_batch[0])
        # mask_src, mask_dst = geo_tools.create_common_region_masks(hom, images_src_batch[0].shape, images_dst_batch[0].shape)
        homography = h_dst_2_src_batch[0].cpu().numpy()
        shape_src = images_src_batch[0].permute(1, 2, 0).cpu().numpy().shape
        shape_dst = images_dst_batch[0].permute(1, 2, 0).cpu().numpy().shape
        mask_src, mask_dst = geometry_tools.create_common_region_masks(homography, shape_src, shape_dst)

        c = time.time()

        src_scores = src_score_maps
        dst_scores = dst_score_maps
        # Apply NMS
        src_scores = repeatability_tools.apply_nms_fast(src_scores[0, 0, :, :].cpu().numpy(), nms_size)
        dst_scores = repeatability_tools.apply_nms_fast(dst_scores[0, 0, :, :].cpu().numpy(), nms_size)

        src_scores = np.multiply(src_scores, mask_src)
        dst_scores = np.multiply(dst_scores, mask_dst)

        d = time.time()

        src_pts = geometry_tools.get_point_coordinates(src_scores, num_points=num_points, order_coord='xysr')
        dst_pts = geometry_tools.get_point_coordinates(dst_scores, num_points=num_points, order_coord='xysr')

        dst_to_src_pts = geometry_tools.apply_homography_to_points(dst_pts, homography)

        e = time.time()

        repeatability_results = repeatability_tools.compute_repeatability(src_pts, dst_to_src_pts)

        rep_s.append(repeatability_results['rep_single_scale'])
        rep_m.append(repeatability_results['rep_multi_scale'])
        error_overlap_s.append(repeatability_results['error_overlap_single_scale'])
        error_overlap_m.append(repeatability_results['error_overlap_multi_scale'])
        possible_matches.append(repeatability_results['possible_matches'])

        f = time.time()

        ## time count
        ba += b-a
        cb += c-b
        dc += d-c
        ed += e-d
        fe += f-e

        iterate.set_description("Validation time: {:0.3f} {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(ba, cb, dc, ed, fe))

    return np.asarray(rep_s).mean(), np.asarray(rep_m).mean(), np.asarray(error_overlap_s).mean(),\
           np.asarray(error_overlap_m).mean(), np.asarray(possible_matches).mean()



def network_outputs_to_score_maps_tensor_batch(network_outputs, cell_size):
    score_maps = F.relu(depth_to_space_without_softmax(network_outputs, cell_size))
    score_maps_np = score_maps.detach().cpu().numpy()

    score_maps_nms_batch = [repeatability_tools.apply_nms_fast(h.squeeze(0), 4) for h in score_maps_np]
    score_maps_nms_batch = np.stack(score_maps_nms_batch, axis=0)
    return torch.tensor(score_maps_nms_batch[:, np.newaxis, ...], dtype=torch.float32)


def train_model(cur_epoch, dataloader, model, optimizer, device, tb_log, tbar, output_dir=None, cell_size=8, add_dustbin=True, anchor_loss='softmax'):
    total_loss_avg = []
    disp_dict = {}

    tic = time.time()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for idx, batch in pbar:
        images_src_batch, images_dst_batch, heatmap_src_patch, heatmap_dst_patch, h_src_2_dst_batch, h_dst_2_src_batch = batch
        # print(im_src_patch.shape, im_dst_patch.shape, heatmap_src_patch.shape, heatmap_dst_patch.shape, homography_src_2_dst.shape, homography_dst_2_src.shape)


        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        tb_log.add_scalar('learning_rate', cur_lr, cur_epoch)


        model.train()
        optimizer.zero_grad()

        src_outputs = model(images_src_batch.to(device))
        dst_outputs = model(images_dst_batch.to(device))

        src_input_heatmaps = space_to_depth(heatmap_src_patch.to(device), cell_size=cell_size, add_dustbin=add_dustbin)
        dst_input_heatmaps = space_to_depth(heatmap_dst_patch.to(device), cell_size=cell_size, add_dustbin=add_dustbin)

        src_anchor_loss = loss_function.anchor_loss(device=device, input=src_outputs, target=src_input_heatmaps, loss_type=anchor_loss)
        dst_anchor_loss = loss_function.anchor_loss(device=device, input=dst_outputs, target=dst_input_heatmaps, loss_type=anchor_loss)

        loss = src_anchor_loss + dst_anchor_loss
        total_loss_avg.append(loss)

        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        pbar.update()
        pbar.set_description('current lr: {:0.6f}, current loss: {:0.4f}, avg loss: {:0.4f}'.format(cur_lr, loss, torch.stack(total_loss_avg).mean()))
        tbar.set_postfix(disp_dict)
        tbar.refresh()

        loss.backward()
        optimizer.step()


        tb_log.add_scalar('train_loss', loss, cur_epoch)

        if idx == 0:
            src_image_grid = torchvision.utils.make_grid(images_src_batch)
            dst_image_grid = torchvision.utils.make_grid(images_dst_batch)
            tb_log.add_image('src_image', src_image_grid, cur_epoch)
            tb_log.add_image('dst_image', dst_image_grid, cur_epoch)

            src_input_heatmaps_grid = torchvision.utils.make_grid(heatmap_src_patch)
            dst_input_heatmaps_grid = torchvision.utils.make_grid(heatmap_dst_patch)
            tb_log.add_image('src_input_heatmap', src_input_heatmaps_grid, cur_epoch)
            tb_log.add_image('dst_input_heatmap', dst_input_heatmaps_grid, cur_epoch)

            src_outputs_score_maps = network_outputs_to_score_maps_tensor_batch(src_outputs, cell_size)
            dst_outputs_score_maps = network_outputs_to_score_maps_tensor_batch(dst_outputs, cell_size)
            src_outputs_score_maps_grid = torchvision.utils.make_grid(src_outputs_score_maps)
            dst_outputs_score_maps_grid = torchvision.utils.make_grid(dst_outputs_score_maps)
            tb_log.add_image('src_output_heatmap', src_outputs_score_maps_grid, cur_epoch)
            tb_log.add_image('dst_output_heatmap', dst_outputs_score_maps_grid, cur_epoch)

        if output_dir is not None and idx % 100 == 0:
            feature_map_dir = str(output_dir) + '/feature_map/'

            feature_map_list = glob.glob(feature_map_dir + '/*.png')
            feature_map_list.sort(key=os.path.getmtime)

            if feature_map_list.__len__() >= 1600:
                for cur_file_idx in range(0, len(feature_map_list) - 1600 + 1):
                    os.remove(feature_map_list[cur_file_idx])

            post_fix = 'epoch'+str(cur_epoch)+'_'+'iter'+str(idx)

            output1 = depth_to_space_without_softmax(src_outputs.detach(), cell_size)
            output2 = depth_to_space_without_softmax(dst_outputs.detach(), cell_size)
            deep_src = common_utils.remove_borders(output1, 16).cpu().detach().numpy()
            deep_dst = common_utils.remove_borders(output2, 16).cpu().detach().numpy()

            common_utils.check_directory(feature_map_dir)

            import matplotlib.pyplot as plt
            fig = plt.figure()
            rows, cols = 2, 2
            ax1 = fig.add_subplot(rows, cols, 1)
            ax1.imshow(images_src_batch[0,:,:,:].permute(1, 2, 0).cpu().detach().numpy())
            ax1.set_title('image_src_' + post_fix)
            ax1.axis("off")

            ax2 = fig.add_subplot(rows, cols, 2)
            ax2.imshow(deep_src[0,0,:,:] / deep_src[0,0,:,:].max(), cmap='gray')
            ax2.set_title('feature_map_src_' + post_fix)
            ax2.axis("off")

            ax3 = fig.add_subplot(rows, cols, 3)
            ax3.imshow(images_dst_batch[0,:,:,:].permute(1, 2, 0).cpu().detach().numpy())
            ax3.set_title('image_dst_' + post_fix)
            ax3.axis("off")

            ax4 = fig.add_subplot(rows, cols, 4)
            ax4.imshow(deep_dst[0,0,:,:] / deep_dst[0,0,:,:].max(), cmap='gray')
            ax4.set_title('feature_map_dst_' + post_fix)
            ax4.axis("off")

            fig.savefig(str(feature_map_dir + post_fix + '.png'))

            # cv2.imwrite(str(feature_map_dir + 'image_src_' + post_fix + '.png'), 255 * images_src_batch[0,:,:,:].permute(1, 2, 0).cpu().detach().numpy())
            # cv2.imwrite(str(feature_map_dir + 'feature_map_src_' + post_fix + '.png'), 255 * deep_src[0,0,:,:] / deep_src[0,0,:,:].max())
            # cv2.imwrite(str(feature_map_dir + 'image_dst_' + post_fix + '.png'), 255 * images_dst_batch[0,:,:,:].permute(1, 2, 0).cpu().detach().numpy())
            # cv2.imwrite(str(feature_map_dir + 'feature_map_dst_' + post_fix + '.png'), 255 * deep_dst[0,0,:,:] / deep_dst[0,0,:,:].max())
            

    toc = time.time()
    total_loss_avg = torch.stack(total_loss_avg)
    logging.info("Epoch {} (Training). Loss: {:0.4f}. Time per epoch: {}".format(cur_epoch, torch.mean(total_loss_avg), round(toc-tic,4)))



def ckpt_state(model=None, optimizer=None, epoch=None, rep_s=0.):
    optim_state = optimizer.state_dict() if optimizer is not None else optimizer
    model_state = model.state_dict() if model is not None else None

    return {'epoch': epoch,  'model_state': model_state, 'optimizer_state': optim_state, 'repeatability': rep_s}