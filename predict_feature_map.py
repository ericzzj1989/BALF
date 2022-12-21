import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import time
import torch
import cv2
import torch.nn as nn

from configs import config_gopro_eval
from utils import common_utils, tensor_op
from utils.logger import logger
from model import get_model
from datasets import dataset_utils
from benchmark_test import test_utils, geometry_tools, repeatability_tools



def get_points_direct_from_score_map_single_image(
    heatmap, conf_thresh=0.015, nms_size=15,
    subpixel=True, patch_size=5, scale_value=1., order_coord='xysr'
):

    H, W = heatmap.shape[0], heatmap.shape[1]
    conf_thresh = 0.01
    ys, xs = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    print('Max reponse: ', np.max(heatmap))
    print('Greater than 0.015 number: ', len(xs))
    if len(xs) == 0:
        return np.zeros((0, 4))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = xs
    pts[1, :] = ys
    pts[2, :] = heatmap[ys, xs]
    pts, _ = repeatability_tools.nms_fast(pts, H, W, dist_thresh=nms_size)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    print('After nms number: ', pts.shape[1])

    if subpixel:
        pts = repeatability_tools.soft_argmax_points(pts, heatmap, patch_size=patch_size)

    print('After subpixel number: ', pts.shape[1])

    new_indexes = []
    for idx in range(pts.shape[1]):
        if order_coord == 'xysr':
            tmp = [pts[0,idx], pts[1,idx], scale_value, pts[2,idx]]
        elif order_coord == 'yxsr':
            tmp = [pts[1,idx], pts[0,idx], scale_value, pts[2,idx]]

        new_indexes.append(tmp)

    indexes = np.asarray(new_indexes)

    return np.asarray(indexes) # N,4

def get_intermediate_results(image_RGB_norm, model, device, args, vis_featuremap_dir):
    height_RGB_norm, width_RGB_norm = image_RGB_norm.shape[0], image_RGB_norm.shape[1]
    
    image_even = dataset_utils.make_shape_even(image_RGB_norm)
    height_even, width_even = image_even.shape[0], image_even.shape[1]
    
    image_pad = dataset_utils.mod_padding_symmetric(image_even, factor=64)
    

    image_pad_tensor = torch.tensor(image_pad, dtype=torch.float32)
    image_pad_tensor = image_pad_tensor.permute(2, 0, 1)
    image_pad_batch = image_pad_tensor.unsqueeze(0)

    t0 = time.time()
    output_pad_batch = model(image_pad_batch.to(device))
    t1 = time.time()
    extract_time = t1-t0


    prob = output_pad_batch['logits']
    softmax = torch.nn.Softmax(dim=1)
    prob = softmax(prob)
    prob = prob[:, :-1, :, :]
    prob = tensor_op.pixel_shuffle(prob, 8)
    score_map_pad_batch = prob.squeeze(dim=1)
    score_map_pad_np = score_map_pad_batch[0, :, :].cpu().detach().numpy()

    # unpad images to get the original resolution
    new_height, new_width = score_map_pad_np.shape[0], score_map_pad_npx.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height_RGB_norm
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width_RGB_norm
    score_map = score_map_pad_np[h_start:h_end, w_start:w_end]


    score_map_remove_border = geometry_tools.remove_borders(score_map, borders=args.border_size)
    score_map_remove_border = score_map_remove_border[np.where(score_map_remove_border>=0.0150)]
    ## method 1 for keypoints
    # score_map_nms = repeatability_tools.apply_nms(score_map_remove_border, 4)
    # print(score_map_remove_border[np.where(score_map_remove_border!=0)])
    cv2.imshow('score_map_nms', score_map_remove_border*255)
    cv2.waitKey(0)

    # if args.output_featuremap:
    #     feature_map = output_pad_batch['logits']
    #     softmax = torch.nn.Softmax(dim=1)
    #     # nn.functional.softmax(input, dim=1)
    #     prob = softmax(feature_map)
    #     feature_map_np = prob[0,:-1,:,:].cpu().detach().numpy() * 255.0
    #     feature_map_np = feature_map_np[:,3:-3,:]
    #     # gray_img = feature_map_np[1,:,:].astype(np.uint8)
    #     # norm_img = np.zeros(gray_img.shape)
    #     # cv2.normalize(gray_img, norm_img,0,255,cv2.NORM_MINMAX)
    #     # norm_img = np.asarray(norm_img, dtype=np.uint8)
    #     # heat_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    #     # heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    #     # plt.figure(1)
    #     # plt.axis("off")
    #     # plt.imshow(heat_img, cmap='jet')
    #     # plt.show()
    #     # jet_fig = plt.gcf()
    #     # cv2.imshow('heat_img', heat_img)
    #     # jet_img = str(vis_featuremap_dir) + '/' + '1_jet.jpg'
    #     # cv2.imwrite(jet_img, heat_img)
    #     # h =  feature_map_np.shape[1]
    #     # w = feature_map_np.shape[2]
    #     # print(h,w)
    #     # cv2.waitKey(0)
    #     for i in range(feature_map_np.shape[0]):
    #         # img = np.zeros((h,w,3), np.uint8)
    #         # ys, xs = np.where(feature_map_np[i,:,:] >= 0.015)
    #         # img[ys,xs, 0] = 0
    #         # img[ys,xs, 1] = 1
    #         # img[ys,xs, 2] = 255
    #         # cv2.imshow('img', img)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #         jet_img = str(vis_featuremap_dir) + '/' + str(i) + '_jet.jpg'
    #         plt.figure(1)
    #         plt.axis("off")
    #         # plt.imshow(feature_map_np[i,:,:], cmap='jet')
    #         # jet_fig = plt.gcf()
    #         # jet_fig.savefig(jet_img, dpi=100, bbox_inches='tight', pad_inches=0)
    #         plt.imsave(fname = jet_img, arr = feature_map_np[i,:,:], dpi = 500, cmap='jet')

    #         # gray_img = str(vis_featuremap_dir) + '/' + str(i) + '_gray.jpg'
    #         # plt.figure(2)
    #         # plt.axis("off")
    #         # plt.imshow(feature_map_np[i,:,:], cmap='gray')
    #         # # gray_fig = plt.gcf()
    #         # # gray_fig.savefig(gray_img, dpi=150, bbox_inches='tight', pad_inches=0)
    #         # plt.imsave(fname = gray_img, arr = feature_map_np[i,:,:], dpi = 1000, cmap='gray')





def main():
    args, cfg = config_gopro_eval.parse_single_image_config()

    output_dir = Path(args.results_detection_dir, args.exper_name)
    common_utils.check_directory(output_dir)
    
    
    logger.initialize(args, output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model.load_model(cfg['model'])
    _,_ = get_model.load_pretrained_model(model=model, filename=args.ckpt_file, logger=logger)
    model.eval()
    model = model.to(device)

    if args.resize_image:
        logger.info('Extract detections in the resized image with {} for repeatability test.'.format(cfg['resize_image']))
    
    image_path = args.image_file

    if not Path(image_path).exists():
        print('[ERROR]: File {0} not found!'.format(image_path))
        return

    logger.info('\nRead image from path: {}'.format(image_path))     

    image_BGR = dataset_utils.read_bgr_image(str(image_path))
    logger.info('Raw image_BGR size: {}'.format(image_BGR.shape))
    if args.resize_image:
        image_BGR = dataset_utils.ratio_preserving_resize(image_BGR, cfg['resize_image'])
    image_RGB = dataset_utils.bgr_to_rgb(image_BGR)
    
    image_RGB_norm = image_RGB / 255.0

    logger.info('Preprocess to image_RGB_norm size: {}'.format(image_RGB_norm.shape))

    if args.is_debugging:
        print('image_RGB_norm size: {}\n'.format(image_RGB_norm.shape))
        plt.figure()
        plt.axis("off")
        plt.imshow(image_RGB_norm)
        plt.show()

    if args.output_featuremap:
        vis_featuremap_dir = Path(output_dir, 'feature_map', Path(image_path).name)
        common_utils.check_directory(vis_featuremap_dir)

    with torch.no_grad():
        get_intermediate_results(image_RGB_norm, model, device, args, vis_featuremap_dir)

if __name__ == '__main__':
    main()