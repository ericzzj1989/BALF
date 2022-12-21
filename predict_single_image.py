import datetime
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from configs import config_gopro_eval
from utils import common_utils
from utils.logger import logger
from model import get_model
from datasets import dataset_utils
from benchmark_test import test_utils, geometry_tools, repeatability_tools



def extract_blur_diff_features(image_RGB_norm, model, device, args, is_debugging=False):
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

    # elif args.nms == 'nms_fast':
    score_map_pad_batch = output_pad_batch['prob']
    score_map_pad_np = score_map_pad_batch[0, :, :].cpu().detach().numpy()

    # unpad images to get the original resolution
    new_height, new_width = score_map_pad_np.shape[0], score_map_pad_np.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height_RGB_norm
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width_RGB_norm
    score_map = score_map_pad_np[h_start:h_end, w_start:w_end]


    score_map_remove_border = geometry_tools.remove_borders(score_map, borders=args.border_size)

    ## method 1 for keypoints
    score_map_nms = repeatability_tools.apply_nms(score_map_remove_border, args.nms_size)

    pts = geometry_tools.get_point_coordinates(score_map_nms, num_points=args.num_points, order_coord='xysr')

    if pts.size == 0:
        return None, None

    pts_sorted = pts[(-1 * pts[:, 3]).argsort()]
    pts_output = pts_sorted[:args.num_points]

    return pts_output, extract_time # pts_output.shape: N*4


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

# def extract_features(image_RGB_norm, model, device, args, vis_featuremap_dir):
def extract_features(image_RGB_norm, model, device, args):
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

    starter , ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions,1))
    with torch.no_grad():
        for _ in range(10):
            _ = model(image_pad_batch.to(device))

    torch.cuda.synchronize()

    with torch.no_grad():
        for rep in tqdm(range(repetitions)):
            starter.record()
            output_pad_batch = model(image_pad_batch.to(device))
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings) / repetitions
    print('this time: {}'.format(mean_syn))

    with profile(activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                model(image_pad_batch.to(device))

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # elif args.nms == 'nms_fast':
    score_map_pad_batch = output_pad_batch['prob']
    score_map_pad_np = score_map_pad_batch[0, :, :].cpu().detach().numpy()

    # unpad images to get the original resolution
    new_height, new_width = score_map_pad_np.shape[0], score_map_pad_np.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height_RGB_norm
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width_RGB_norm
    score_map = score_map_pad_np[h_start:h_end, w_start:w_end]


    score_map_remove_border = geometry_tools.remove_borders(score_map, borders=args.border_size)

    ## method 2 for keypoints
    pts = get_points_direct_from_score_map_single_image(
        heatmap=score_map_remove_border, conf_thresh=args.heatmap_confidence_threshold,
        nms_size=args.nms_size, subpixel=args.sub_pixel, patch_size=args.patch_size, order_coord=args.order_coord
    )

    if pts.size == 0:
        return None, None

    pts_sorted = pts[(-1 * pts[:, 3]).argsort()]
    pts_output = pts_sorted[:args.num_points]

    return pts_output, extract_time # pts_output.shape: N*4




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


    with torch.no_grad():
        image_pts, extract_time = extract_features(image_RGB_norm, model, device, args)

    if image_pts is None and extract_time is None:
        logger.info('No kpts in {}'.format(image_path))
        exit()

    if args.output_detection_img:
        detection_im_src_BGR = test_utils.draw_keypoints(image_BGR, image_pts[:,:2], radius=3)
        visualization_dir = Path(output_dir, 'vis')
        common_utils.check_directory(visualization_dir)
        vis_file = str(visualization_dir)+'/'+str(Path(image_path).stem)+'.svg'

        plt.figure(figsize=(8.25, 6.18), dpi=600, facecolor='w', edgecolor='k')
        plt.axis("off")
        detection_im_src_RGB = dataset_utils.bgr_to_rgb(detection_im_src_BGR)
        plt.imshow(detection_im_src_RGB)
        img_det_plt = plt.gcf()
        img_det_plt.savefig(vis_file, bbox_inches='tight', pad_inches=0)

        # test_utils.plot_imgs(
        #     [detection_im_src_BGR.astype(np.uint8)],
        #      titles=['pts: {}'.format(image_pts.shape[0])], dpi=300)
        # plt.tight_layout()
        # plt.savefig(vis_file, dpi=300, bbox_inches='tight')

        # vis_file = str(visualization_dir)+'/'+str(Path(image_path).name)
        # cv2.imwrite(vis_file, detection_im_src_BGR)
    
    if args.save_kpts:
        detection_output_dir = output_dir / 'detection'
        common_utils.check_directory(detection_output_dir)
        save_pts_dir = Path(detection_output_dir, Path(image_path).name)
        common_utils.create_result_dir(str(save_pts_dir))
        kpt_file = Path(str(save_pts_dir)+'.kpt')
        np.savez(kpt_file, kpts=image_pts)
        logger.info('{} kpts saved in {}\n'.format(image_pts.shape, kpt_file))

    logger.info('{} kpts extracted, extract_time {:.5f} in image size {}'.format(image_pts.shape, extract_time, image_BGR.shape[:2]))
    logger.info("================ End Extractiton ================ \n")

if __name__ == '__main__':
    main()