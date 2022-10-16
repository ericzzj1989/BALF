import os
import logging
import time
import numpy as np
import math
import glob
from tqdm import tqdm
import cv2

import torch
import torch.optim as optim
import torchvision

from benchmark_test import repeatability_tools, geometry_tools
from loss import loss_function
from utils import common_utils
from datasets import dataset_utils


def build_optimizer(params, optim_cfg):
    logging.info("=> Setting %s solver", optim_cfg['name'])

    if optim_cfg['name'] == 'adam':
        optimizer = optim.Adam(params, lr=optim_cfg['lr'], weight_decay=optim_cfg['weight_decay'])
    elif optim_cfg['name'] == 'sgd':
        optimizer = optim.SGD(params, lr=optim_cfg['lr'])
    else:
        raise ValueError("Optimizer [%s] not recognized." % optim_cfg['name'])
    return optimizer


def build_scheduler(optimizer, total_epochs, last_epoch, scheduler_cfg):
    logging.info("=> Setting %s scheduler", scheduler_cfg['name'])
    if scheduler_cfg['name'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            patience=scheduler_cfg['patience'],
                                                            factor=scheduler_cfg['factor'],
                                                            min_lr=scheduler_cfg['min_lr'])
    elif scheduler_cfg['name'] == 'sgdr':
        scheduler = WarmRestart(optimizer)
    elif scheduler_cfg['name'] == 'linear':
        scheduler = LinearDecay(optimizer,
                                min_lr=scheduler_cfg['min_lr'],
                                max_epoch=total_epochs,
                                start_epoch=scheduler_cfg['start_epoch'],
                                last_epoch=last_epoch)
    else:
        raise ValueError("Scheduler [%s] not recognized." % scheduler_cfg['name'])
    return scheduler

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


class LinearDecay(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epoch, start_epoch=0, min_lr=0, last_epoch=-1):
        self.max_epoch = max_epoch
        self.start_epoch = start_epoch
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.start_epoch:
            return self.base_lrs
        return [base_lr - ((base_lr - self.min_lr) / self.max_epoch) * (self.last_epoch - self.start_epoch) for
                base_lr in self.base_lrs]




def train_model(
    cur_epoch, dataloader, model, optimizer, device, tb_log,
    tbar, output_dir=None, cell_size=8, add_dustbin=True,
    anchor_loss='softmax', usp_loss=None, repeatability_loss=None):

    total_loss_avg = []
    disp_dict = {}

    tic = time.time()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for idx, batch in pbar:
        images_src_batch, images_dst_batch, heatmap_src_patch, heatmap_dst_patch, h_src_2_dst_batch, h_dst_2_src_batch = batch
        # print(im_src_patch.shape, im_dst_patch.shape, heatmap_src_patch.shape, heatmap_dst_patch.shape, homography_src_2_dst.shape, homography_dst_2_src.shape)
        images_src_batch, images_dst_batch, heatmap_src_patch, heatmap_dst_patch, h_src_2_dst_batch, h_dst_2_src_batch =\
        images_src_batch.to(device), images_dst_batch.to(device), heatmap_src_patch.to(device), heatmap_dst_patch.to(device), h_src_2_dst_batch.to(device), h_dst_2_src_batch.to(device)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        tb_log.add_scalar('learning_rate', cur_lr, cur_epoch)


        model.train()
        

        src_outputs  = model(images_src_batch)
        dst_outputs = model(images_dst_batch)

        '''
        src_input_heatmaps_batch = space_to_depth(heatmap_src_patch, cell_size=cell_size, add_dustbin=add_dustbin)
        dst_input_heatmaps_batch = space_to_depth(heatmap_dst_patch, cell_size=cell_size, add_dustbin=add_dustbin)

        src_anchor_loss = loss_function.anchor_loss(
            device=device, input=src_outputs['logits'],
            target=src_input_heatmaps_batch, loss_type=anchor_loss
        )
        dst_anchor_loss = loss_function.anchor_loss(
            device=device, input=dst_outputs['logits'],
            target=dst_input_heatmaps_batch, loss_type=anchor_loss
        )
        '''

        src_anchor_loss = loss_function.detector_loss(
            keypoint_map=heatmap_src_patch,
            logits=src_outputs['logits'],
            device=device
        )
        dst_anchor_loss = loss_function.detector_loss(
            keypoint_map=heatmap_dst_patch,
            logits=dst_outputs['logits'],
            device=device
        )


        '''if usp_loss is not None:
            src_flags = heatmap_flag(heatmap_src_patch, cell_size=cell_size)
            dst_flags = heatmap_flag(heatmap_dst_patch, cell_size=cell_size)
            unsuper_loss, unsuper_loss_item = usp_loss.loss(
                src_outputs_score_batch, dst_outputs_score_batch,
                src_outputs_pos_batch, dst_outputs_pos_batch,
                h_src_2_dst_batch, h_dst_2_src_batch,
                cell_size, src_flags, dst_flags
            )
        else:
            unsuper_loss = torch.tensor(0.0).float().to(device)

        if repeatability_loss is not None:
            shape_src = images_src_batch.shape[2:]
            shape_dst = images_dst_batch.shape[2:]
            rep_loss = repeatability_loss['weight'] * loss_function.repeatability_loss_batch(
                src_outputs_score_batch, dst_outputs_score_batch, h_dst_2_src_batch, shape_src, shape_dst, cell_size, 15, 25
            )
        else:
            rep_loss = torch.tensor(0.0).float().to(device)'''


        loss = src_anchor_loss + dst_anchor_loss # + unsuper_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_avg.append(loss)

        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})
        pbar.update()
        pbar.set_description('Training current lr: {:0.6f}, current loss: {:0.4f}, avg loss: {:0.4f}'.format(cur_lr, loss, torch.stack(total_loss_avg).mean()))
        tbar.set_postfix(disp_dict)
        tbar.refresh()

        if idx == 0:
            tb_log.add_scalar('total_loss', loss, cur_epoch)
            tb_log.add_scalar('src_anchor_loss', src_anchor_loss, cur_epoch)
            tb_log.add_scalar('dst_anchor_loss', dst_anchor_loss, cur_epoch)
            # tb_log.add_scalar('unsuper_loss', unsuper_loss, cur_epoch)
            # tb_log.add_scalar('usp_loss', ussuper_loss_item[0], cur_epoch)
            # tb_log.add_scalar('uni_loss', ussuper_loss_item[1], cur_epoch)
            # tb_log.add_scalar('rep_loss', rep_loss, cur_epoch)

            src_image_grid = torchvision.utils.make_grid(images_src_batch)
            dst_image_grid = torchvision.utils.make_grid(images_dst_batch)
            tb_log.add_image('train_src_image', src_image_grid, cur_epoch)
            tb_log.add_image('train_dst_image', dst_image_grid, cur_epoch)

            src_input_heatmaps_grid = torchvision.utils.make_grid(heatmap_src_patch)
            dst_input_heatmaps_grid = torchvision.utils.make_grid(heatmap_dst_patch)
            tb_log.add_image('train_src_input_heatmap', src_input_heatmaps_grid, cur_epoch)
            tb_log.add_image('train_dst_input_heatmap', dst_input_heatmaps_grid, cur_epoch)

            src_outputs_score_maps = prob_to_score_maps_tensor_batch(src_outputs['prob'], 4)
            dst_outputs_score_maps = prob_to_score_maps_tensor_batch(dst_outputs['prob'], 4)
            src_outputs_score_maps_grid = torchvision.utils.make_grid(src_outputs_score_maps)
            dst_outputs_score_maps_grid = torchvision.utils.make_grid(dst_outputs_score_maps)
            tb_log.add_image('train_src_output_heatmap', src_outputs_score_maps_grid, cur_epoch)
            tb_log.add_image('train_dst_output_heatmap', dst_outputs_score_maps_grid, cur_epoch)

        if output_dir is not None and idx % 20 == 0:
            feature_map_dir = str(output_dir) + '/feature_map/'

            feature_map_list = glob.glob(feature_map_dir + '/*.png')
            feature_map_list.sort(key=os.path.getmtime)

            if feature_map_list.__len__() >= 1600:
                for cur_file_idx in range(0, len(feature_map_list) - 1600 + 1):
                    os.remove(feature_map_list[cur_file_idx])

            post_fix = 'epoch'+str(cur_epoch)+'_'+'iter'+str(idx)

            output1 = src_outputs['prob'].unsqueeze(1).detach()
            output2 = dst_outputs['prob'].unsqueeze(1).detach()
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
    logging.info('Epoch {} (Training). Loss: {:0.4f}. Time per epoch: {}. \n'.format(cur_epoch, torch.mean(total_loss_avg), round(toc-tic,4)))


def prob_to_score_maps_tensor_batch(prob_outputs, nms_size):
    score_maps_np = prob_outputs.detach().cpu().numpy()

    score_maps_nms_batch = [repeatability_tools.apply_nms_fast(h, nms_size) for h in score_maps_np]
    score_maps_nms_batch = np.stack(score_maps_nms_batch, axis=0)
    score_maps_nms_batch_tensor = torch.tensor(score_maps_nms_batch, dtype=torch.float32)
    return score_maps_nms_batch_tensor.unsqueeze(1)


def compute_repeatability_with_maximum_filter(src_scores_np, dst_scores_np, homography, mask_src, mask_dst, nms_size, num_points):
    rep_s_nms = []
    rep_m_nms = []
    error_overlap_s_nms = []
    error_overlap_m_nms = []
    possible_matches_nms = []

    src_scores_nms = repeatability_tools.apply_nms(src_scores_np, nms_size)
    dst_scores_nms = repeatability_tools.apply_nms(dst_scores_np, nms_size)

    src_scores_common_nms = np.multiply(src_scores_nms, mask_src)
    dst_scores_common_nms = np.multiply(dst_scores_nms, mask_dst)

    src_pts_nms = geometry_tools.get_point_coordinates(src_scores_common_nms, num_points=num_points, order_coord='xysr')
    dst_pts_nms = geometry_tools.get_point_coordinates(dst_scores_common_nms, num_points=num_points, order_coord='xysr')

    dst_to_src_pts_nms = geometry_tools.apply_homography_to_points(dst_pts_nms, homography)

    repeatability_results_nms = repeatability_tools.compute_repeatability(src_pts_nms, dst_to_src_pts_nms)

    rep_s_nms.append(repeatability_results_nms['rep_single_scale'])
    rep_m_nms.append(repeatability_results_nms['rep_multi_scale'])
    error_overlap_s_nms.append(repeatability_results_nms['error_overlap_single_scale'])
    error_overlap_m_nms.append(repeatability_results_nms['error_overlap_multi_scale'])
    possible_matches_nms.append(repeatability_results_nms['possible_matches'])

    return rep_s_nms, rep_m_nms, error_overlap_s_nms, error_overlap_m_nms, possible_matches_nms


'''
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
'''


'''
class DepthToSpace(torch.nn.Module):
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

class SpaceToDepth(torch.nn.Module):
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

def depth_to_space_with_softmax(score_maps, cell_size):
    is_batch = False
    if len(score_maps.shape) == 4:
        is_batch = True

    if is_batch:
        maps = torch.nn.functional.softmax(score_maps, dim=1) # [batch, 65, Hc, Wc]
        nodust = maps[:, :-1, :, :]
    else:
        maps = torch.nn.functional.softmax(score_maps, dim=0) # [65, Hc, Wc]
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
        if score_maps.shape[1] == 65:
            nodust = score_maps[:, :-1, :, :]
        elif score_maps.shape[1] == 64:
            nodust = score_maps
    else:
        if score_maps.shape[0] == 65:
            nodust = score_maps[:-1, :, :].unsqueeze(0)
        elif score_maps.shape[0] == 64:
            nodust = score_maps.unsqueeze(0)

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
        heatmaps = heatmaps.div(torch.unsqueeze(dn, 1))
    
    score_maps = heatmaps.squeeze(0) if not is_batch else heatmaps
    return score_maps # [batch, 65, Hc, Wc]

def heatmap_flag(heatmaps, cell_size=8):
    is_batch = True
    if len(heatmaps.shape) == 3:
        is_batch = False
        heatmaps = heatmaps.unsqueeze(0)

    batch_size, _, H, W = heatmaps.shape # labels [batch, 1, H, W]
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(cell_size)
    heatmaps = space2depth(heatmaps)
    # if add_dustbin:
    dustbin = heatmaps.sum(dim=1)
    dustbin = 1 - dustbin
    dustbin[dustbin < 1.] = 0
    dustbin = 1 - dustbin
    flag = dustbin.view(batch_size, 1, Hc, Wc)
    
    flags = flag.squeeze(0) if not is_batch else flag
    flags = flags.type(torch.bool)
    return flags # [batch, 1, Hc, Wc]
'''






'''
def get_position(p_map, downsample=8):
    x = torch.arange(p_map.shape[2], requires_grad=False, device=p_map.device)
    y = torch.arange(p_map.shape[1], requires_grad=False, device=p_map.device)
    y, x = torch.meshgrid([y, x])
    cell = torch.stack([x, y], dim=0)

    return (cell + p_map) * downsample

def get_score_map(score_map, position, downsample=8): # score_map: 1,512,512, position: 2 32 32
    pixel_coor = get_position(position, downsample).to(position.device)

    pixel_coor[torch.where(pixel_coor>=(score_map.shape[1]-0.5))] = score_map.shape[1]-1.0

    score = score_map[:, pixel_coor[1,:,:].round().long(), pixel_coor[0,:,:].round().long()]

    assert(score.shape[1] == (score_map.shape[1] // downsample) and score.shape[2] == (score_map.shape[2] // downsample))

    return score # score: 1,32,32

def get_coordinates(positions, scores, scale_value=1., num_points=1000, threshold=-1, order_coord='xysr'):
    scores = scores.squeeze(0)
    indexes = geometry_tools.find_index_higher_scores(scores, num_points=num_points, threshold=threshold)
    new_indexes = []
    for ind in indexes:
        position = positions[:,ind[0],ind[1]]
        score = scores[ind[0],ind[1]]
        if order_coord == 'xysr':
            tmp = [position[0], position[1], scale_value, score]
        elif order_coord == 'yxsr':
            tmp = [position[1], position[0], scale_value, score]

        new_indexes.append(tmp)

    indexes = np.asarray(new_indexes)

    return np.asarray(indexes)

def compute_repeatability_with_direct_position(
    src_score_maps, src_outputs_pos_batch,
    dst_score_maps, dst_outputs_pos_batch, homography,
    num_points, order_coord):
    
    rep_s = []
    rep_m = []
    error_overlap_s = []
    error_overlap_m = []
    possible_matches = []


    src_positions = get_position(src_outputs_pos_batch[0,:,:,:]) # 2 32 32
    src_scores = get_score_map(src_score_maps[0,:,:,:], src_outputs_pos_batch[0,:,:,:]) # 32,32

    src_pts = get_coordinates(src_positions.cpu().numpy(), src_scores.cpu().numpy(), num_points=num_points, order_coord=order_coord)

    dst_positions = get_position(dst_outputs_pos_batch[0,:,:,:]) # 2 32 32
    dst_scores = get_score_map(dst_score_maps[0,:,:,:], dst_outputs_pos_batch[0,:,:,:]) # 32,32

    dst_pts = get_coordinates(dst_positions.cpu().numpy(), dst_scores.cpu().numpy(), num_points=num_points, order_coord=order_coord)

    dst_to_src_pts = geometry_tools.apply_homography_to_points(dst_pts, homography)

    repeatability_results = repeatability_tools.compute_repeatability(src_pts, dst_to_src_pts)

    rep_s.append(repeatability_results['rep_single_scale'])
    rep_m.append(repeatability_results['rep_multi_scale'])
    error_overlap_s.append(repeatability_results['error_overlap_single_scale'])
    error_overlap_m.append(repeatability_results['error_overlap_multi_scale'])
    possible_matches.append(repeatability_results['possible_matches'])

    return rep_s, rep_m, error_overlap_s, error_overlap_m, possible_matches
'''






def ckpt_state(model=None, optimizer=None, epoch=None, rep_s=0.):
    optim_state = optimizer.state_dict() if optimizer is not None else optimizer
    model_state = model.state_dict() if model is not None else None

    return {'epoch': epoch,  'model_state': model_state, 'optimizer_state': optim_state, 'repeatability': rep_s}




@torch.no_grad()
def check_val_repeatability(dataloader, model, device, tb_log, cur_epoch, cell_size=8, nms_size=15, num_points=25):
    rep_s = []
    rep_m = []
    error_overlap_s = []
    error_overlap_m = []
    possible_matches = []
    disp_dict = {}

    model.eval()

    iterate = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")

    ba = 0; cb =0; dc= 0; ed= 0; fe=0
    for idx, batch in iterate:
        a = time.time()
        images_src_batch, images_dst_batch, heatmap_src_patch, heatmap_dst_patch, h_src_2_dst_batch, h_dst_2_src_batch = batch
        images_src_batch, images_dst_batch, heatmap_src_patch, heatmap_dst_patch, h_src_2_dst_batch, h_dst_2_src_batch =\
        images_src_batch.to(device), images_dst_batch.to(device), heatmap_src_patch.to(device), heatmap_dst_patch.to(device), h_src_2_dst_batch.to(device), h_dst_2_src_batch.to(device)

        src_outputs = model(images_src_batch)
        dst_outputs = model(images_dst_batch)

        src_score_maps = src_outputs['prob']
        dst_score_maps = dst_outputs['prob']

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
        src_scores = repeatability_tools.get_nms_score_map_from_score_map(src_scores[0, :, :].cpu().numpy(), nms_size)
        dst_scores = repeatability_tools.get_nms_score_map_from_score_map(dst_scores[0, :, :].cpu().numpy(), nms_size)

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

        iterate.update()
        iterate.set_description('Validation time: {:0.3f} {:0.3f} {:0.3f} {:0.3f} {:0.3f}'.format(ba, cb, dc, ed, fe))
        disp_dict.update({'rep_s': '{:0.2f}'.format(repeatability_results['rep_single_scale'])})
        iterate.set_postfix(disp_dict)

        rep_s_nms, rep_m_nms, error_overlap_s_nms, error_overlap_m_nms, possible_matches_nms =\
            compute_repeatability_with_maximum_filter(
                src_score_maps[0, :, :].cpu().numpy(), dst_score_maps[0, :, :].cpu().numpy(), homography,
                mask_src, mask_dst, nms_size, num_points
        )
        
        # rep_s_pos, rep_m_pos, error_overlap_s_pos, error_overlap_m_pos, possible_matches_pos = compute_repeatability_with_direct_position(
        #     src_score_maps, src_outputs_pos_batch, dst_score_maps, dst_outputs_pos_batch, homography,
        #     num_points=num_points, order_coord='xysr'
        # )

        if tb_log is not None and idx == 0:
            src_image_grid = torchvision.utils.make_grid(images_src_batch)
            dst_image_grid = torchvision.utils.make_grid(images_dst_batch)
            tb_log.add_image('val_src_image', src_image_grid, cur_epoch)
            tb_log.add_image('val_dst_image', dst_image_grid, cur_epoch)
            
            src_input_heatmaps_grid = torchvision.utils.make_grid(heatmap_src_patch)
            dst_input_heatmaps_grid = torchvision.utils.make_grid(heatmap_dst_patch)
            tb_log.add_image('val_src_input_heatmap', src_input_heatmaps_grid, cur_epoch)
            tb_log.add_image('val_dst_input_heatmap', dst_input_heatmaps_grid, cur_epoch)

            src_outputs_score_maps = prob_to_score_maps_tensor_batch(src_outputs['prob'], nms_size)
            dst_outputs_score_maps = prob_to_score_maps_tensor_batch(dst_outputs['prob'], nms_size)
            src_outputs_score_maps_grid = torchvision.utils.make_grid(src_outputs_score_maps)
            dst_outputs_score_maps_grid = torchvision.utils.make_grid(dst_outputs_score_maps)
            tb_log.add_image('val_src_output_heatmap', src_outputs_score_maps_grid, cur_epoch)
            tb_log.add_image('val_dst_output_heatmap', dst_outputs_score_maps_grid, cur_epoch)
            
    return np.asarray(rep_s).mean(), np.asarray(rep_m).mean(), np.asarray(error_overlap_s).mean(),\
           np.asarray(error_overlap_m).mean(), np.asarray(possible_matches).mean(),\
           np.asarray(rep_s_nms).mean(), np.asarray(rep_m_nms).mean(), np.asarray(error_overlap_s_nms).mean(),\
           np.asarray(error_overlap_m_nms).mean(), np.asarray(possible_matches_nms).mean() #,\
        #    np.asarray(rep_s_pos).mean(), np.asarray(rep_m_pos).mean(), np.asarray(error_overlap_s_pos).mean(),\
        #    np.asarray(error_overlap_m_pos).mean(), np.asarray(possible_matches_pos).mean()


@torch.no_grad()
def check_val_hsequences_repeatability(dataloader, model, device, tb_log, cur_epoch, cell_size=8, nms_size=15, num_points=25, border_size=15):
    rep_s = []
    rep_m = []
    error_overlap_s = []
    error_overlap_m = []
    possible_matches = []
    disp_dict = {}
    
    counter_sequences = 0
    iterate = tqdm(range(len(dataloader.sequences)), total=len(dataloader.sequences), desc="HSequences Eval")

    ba = 0; cb =0; dc= 0; ed= 0; fe=0
    for sequence_index in iterate:
        sequence_data = dataloader.get_sequence_data(sequence_index)

        counter_sequences += 1

        sequence_name = sequence_data['sequence_name']
        im_src_RGB_norm = sequence_data['im_src_RGB_norm']
        images_dst_RGB_norm = sequence_data['images_dst_RGB_norm']
        # h_src_2_dst = sequence_data['h_src_2_dst']
        h_dst_2_src = sequence_data['h_dst_2_src']

        for im_dst_index in range(len(images_dst_RGB_norm)):
            a = time.time()
            pts_src, src_outputs_score_maps = extract_detections(
                im_src_RGB_norm, model, device,
                cell_size=cell_size, nms_size=nms_size, num_points=num_points, border_size=border_size
            )
            pts_dst, dst_outputs_score_maps = extract_detections(
                images_dst_RGB_norm[im_dst_index], model, device,
                cell_size=cell_size, nms_size=nms_size, num_points=num_points, border_size=border_size
            )

            b = time.time()

            mask_src, mask_dst = geometry_tools.create_common_region_masks(
                h_dst_2_src[im_dst_index], im_src_RGB_norm.shape, images_dst_RGB_norm[im_dst_index].shape
            )

            c = time.time()

            
            pts_src = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_src)))
            pts_dst = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_dst)))
            
            idx_src = repeatability_tools.check_common_points(pts_src, mask_src)
            if idx_src.size == 0:
                continue
            pts_src = pts_src[idx_src]

            idx_dst = repeatability_tools.check_common_points(pts_dst, mask_dst)
            if idx_dst.size == 0:
                continue
            pts_dst = pts_dst[idx_dst]

            d = time.time()

            pts_src = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_src)))
            pts_dst = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], pts_dst)))

            pts_dst_to_src = geometry_tools.apply_homography_to_points(
                pts_dst, h_dst_2_src[im_dst_index])

            e = time.time()

            repeatability_results = repeatability_tools.compute_repeatability(pts_src, pts_dst_to_src)

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

            
            iterate.set_description("{}  {} / {} - {} Validation time: {:0.3f} {:0.3f} {:0.3f} {:0.3f} {:0.3f}"
                .format(
                    sequence_name, counter_sequences, len(dataloader.sequences), im_dst_index,
                    ba, cb, dc, ed, fe
            ))
            disp_dict.update({'rep_s': '{:0.2f}'.format(repeatability_results['rep_single_scale'])})
            iterate.set_postfix(disp_dict)

            if tb_log is not None and sequence_index == 50:
                im_src_RGB_norm_tensor = torch.tensor(im_src_RGB_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                im_dst_RGB_norm_tensor = torch.tensor(images_dst_RGB_norm[im_dst_index], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                src_image_grid = torchvision.utils.make_grid(im_src_RGB_norm_tensor)
                dst_image_grid = torchvision.utils.make_grid(im_dst_RGB_norm_tensor)
                tb_log.add_image('val_src_image', src_image_grid, cur_epoch)
                tb_log.add_image('val_dst_image', dst_image_grid, cur_epoch)

                src_outputs_score_maps_grid = torchvision.utils.make_grid(src_outputs_score_maps)
                dst_outputs_score_maps_grid = torchvision.utils.make_grid(dst_outputs_score_maps)
                tb_log.add_image('val_src_output_heatmap', src_outputs_score_maps_grid, cur_epoch)
                tb_log.add_image('val_dst_output_heatmap', dst_outputs_score_maps_grid, cur_epoch)


    return np.asarray(rep_s).mean(), np.asarray(rep_m).mean(), np.asarray(error_overlap_s).mean(),\
           np.asarray(error_overlap_m).mean(), np.asarray(possible_matches).mean()


@torch.no_grad()
def extract_detections(image_RGB_norm, model, device, cell_size=8, nms_size=15, num_points=25, border_size=15): 
    height_RGB_norm, width_RGB_norm = image_RGB_norm.shape[0], image_RGB_norm.shape[1]
    
    image_even = dataset_utils.make_shape_even(image_RGB_norm)
    height_even, width_even = image_even.shape[0], image_even.shape[1]
    
    image_pad = dataset_utils.mod_padding_symmetric(image_even, factor=64)
    

    image_pad_tensor = torch.tensor(image_pad, dtype=torch.float32)
    image_pad_tensor = image_pad_tensor.permute(2, 0, 1)
    image_pad_batch = image_pad_tensor.unsqueeze(0)

    with torch.no_grad():
        output_pad_batch = model(image_pad_batch.to(device))

    score_map_pad_batch = output_pad_batch['prob']
    score_map_pad_np = score_map_pad_batch[0, :, :].cpu().detach().numpy()

    # unpad images to get the original resolution
    new_height, new_width = score_map_pad_np.shape[0], score_map_pad_np.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height_RGB_norm
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width_RGB_norm
    score_map = score_map_pad_np[h_start:h_end, w_start:w_end]

    score_map_batch = score_map_pad_batch[:,:,h_start:h_end, w_start:w_end]

    score_map_remove_border = geometry_tools.remove_borders(score_map, borders=border_size)
    score_map_nms = repeatability_tools.apply_nms(score_map_remove_border, nms_size)

    pts = geometry_tools.get_point_coordinates(score_map_nms, num_points=num_points, order_coord='xysr')

    pts_sorted = pts[(-1 * pts[:, 3]).argsort()]
    pts_output = pts_sorted[:num_points]

    return pts_output, score_map_batch