import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import time
import torch

from configs import config_gopro_eval
from utils import common_utils
from utils.logger import logger
from model import get_model
from datasets import dataset_utils
from benchmark_test import geometry_tools, repeatability_tools


def extract_features(image_RGB_norm, model, device, args, is_debugging=False):
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


    if args.nms == 'apply_nms':
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
        if is_debugging:
            dataset_utils.debug_test_results(
                image_RGB_norm, image_even, image_pad, score_map_pad_np,
                score_map, score_map_remove_border, score_map_nms
            )
        pts = geometry_tools.get_point_coordinates(score_map_nms, num_points=args.num_points, order_coord='xysr')

    elif args.nms == 'apply_nms_fast':
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

        score_map_remove_border_nms = repeatability_tools.apply_nms_fast(score_map_remove_border, args.nms_size, args.heatmap_confidence_threshold)

        keypoints = np.where(score_map_remove_border_nms > 0)
        prob = score_map_remove_border_nms[keypoints[0], keypoints[1]]
        keypoints = np.stack([keypoints[1],
                              keypoints[0],
                              prob], axis=-1)

        kpts = keypoints.transpose()

        inds = np.argsort(kpts[2, :])
        kpts = kpts[:, inds[::-1]]
        
        if args.sub_pixel:
            kpts = repeatability_tools.soft_argmax_points(kpts, score_map_remove_border_nms, patch_size=args.patch_size)

        new_indexes = []
        for idx in range(kpts.shape[1]):
            if args.order_coord == 'xysr':
                tmp = [kpts[0,idx], kpts[1,idx], 1., kpts[2,idx]]
            elif args.order_coord == 'yxsr':
                tmp = [kpts[1,idx], kpts[0,idx], 1., kpts[2,idx]]

            new_indexes.append(tmp)

        pts = np.asarray(new_indexes)

    elif args.nms == 'nms_fast':
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
        pts = repeatability_tools.get_points_direct_from_score_map(
            heatmap=score_map_remove_border, conf_thresh=args.heatmap_confidence_threshold,
            nms_size=args.nms_size, subpixel=args.sub_pixel, patch_size=args.patch_size, order_coord=args.order_coord
        )

    elif args.nms == 'box_nms':
        score_map_pad_batch = score_map_pad_batch = output_pad_batch['prob']
        # score_map_pad_np = score_map_pad_batch[0, 0, :, :].cpu().detach().numpy()

        # unpad images to get the original resolution
        new_height, new_width = score_map_pad_batch.shape[2], score_map_pad_batch.shape[3]
        h_start = new_height // 2 - height_even // 2
        h_end = h_start + height_RGB_norm
        w_start = new_width // 2 - width_even // 2
        w_end = w_start + width_RGB_norm
        score_map_tensor = score_map_pad_batch[:,h_start:h_end, w_start:w_end]


        score_map_remove_border_tensor = geometry_tools.remove_borders_tensor(score_map_tensor, borders=args.border_size)

        prob_nms_tensor = repeatability_tools.box_nms(score_map_remove_border_tensor[0,:,:].unsqueeze(dim=0),
                        size=args.nms_size,
                        iou=0.3,
                        min_prob=args.heatmap_confidence_threshold,
                        keep_top_k=args.num_points).squeeze(dim=0)

        prob_nms_np = prob_nms_tensor.cpu().detach().numpy()

        keypoints = np.where(prob_nms_np > 0)
        prob = prob_nms_np[keypoints[0], keypoints[1]]
        keypoints = np.stack([keypoints[1],
                              keypoints[0],
                              prob], axis=-1)

        kpts = keypoints.transpose()

        inds = np.argsort(kpts[2, :])
        kpts = kpts[:, inds[::-1]]
        
        if args.sub_pixel:
            kpts = repeatability_tools.soft_argmax_points(kpts, prob_nms_np, patch_size=args.patch_size)

        new_indexes = []
        for idx in range(kpts.shape[1]):
            if args.order_coord == 'xysr':
                tmp = [kpts[0,idx], kpts[1,idx], 1., kpts[2,idx]]
            elif args.order_coord == 'yxsr':
                tmp = [kpts[1,idx], kpts[0,idx], 1., kpts[2,idx]]

            new_indexes.append(tmp)

        pts = np.asarray(new_indexes)




    if pts.size == 0:
        return None, None

    pts_sorted = pts[(-1 * pts[:, 3]).argsort()]
    pts_output = pts_sorted[:args.num_points]

    return pts_output, extract_time # pts_output.shape: N*4



def main():
    args, cfg = config_gopro_eval.parse_config()

    assert(args.comparison_method in args.exper_name)
    assert(args.comparison_method in args.gopro_list_file)

    if args.comparison_method == 'src_sharp_dst_blur':
        args.results_detection_dir = 'results_gopro_src_sharp_dst_blur_detection'
    elif args.comparison_method == 'src_blur_dst_sharp':
        args.num_points = 2000
        args.nms = 'apply_nms'
        args.results_detection_dir = 'results_gopro_src_blur_dst_sharp_detection'
    elif args.comparison_method == 'src_blur_dst_blur':
        args.num_points = 2000
        args.nms = 'apply_nms'
        args.results_detection_dir = 'results_gopro_src_blur_dst_blur_detection'
    elif args.comparison_method == 'src_blur_dst_blur_diff':
        args.num_points = 3000
        args.nms = 'apply_nms'
        args.results_detection_dir = 'results_gopro_src_blur_dst_blur_diff_detection'

    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = Path(args.results_detection_dir, args.exper_name, start_time)
    common_utils.check_directory(output_dir)
    detection_output_dir = output_dir / 'detection'
    common_utils.check_directory(detection_output_dir)
    
    logger.initialize(args, output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model.load_model(cfg['model'])
    _,_ = get_model.load_pretrained_model(model=model, filename=args.ckpt_file, logger=logger)
    model.eval()
    model = model.to(device)

    if args.resize_image:
        logger.info('Extract detections in the resized image with {} for repeatability test.'.format(cfg['resize_image']))

    total_time = []
    counter_fail_extraction = 0
    
    list_file = open(args.gopro_list_file, 'r')
    images_list = sorted(list_file.readlines())
    iterate = tqdm(images_list, total=len(images_list), desc="Motion Blur Detection")

    for path_to_image in iterate:
        image_path = path_to_image.rstrip('\n')
        iterate.set_description("Current {}".format(image_path))

        if not Path(image_path).exists():
            print('[ERROR]: File {0} not found!'.format(image_path))
            return

        logger.info("================ Strat Extractiton ================ ")
        logger.info('Read image from path: {}'.format(image_path))     

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
            logger.info('No kpts in {}'.format(path_to_image.rstrip('\n')))
            counter_fail_extraction += 1
            continue

        logger.info('Save image path: {}'.format(image_path))

        save_pts_dir = Path(detection_output_dir, image_path)
        common_utils.create_result_dir(str(save_pts_dir))
        
        kpt_file = Path(str(save_pts_dir)+'.kpt')

        assert image_path.split('/')[-1] == str(kpt_file).split('/')[-1][:-4]

        np.savez(kpt_file, kpts=image_pts)

        total_time.append(extract_time)

        logger.info('{} kpts saved in {}, extract_time {:.5f} in image size {}'.format(image_pts.shape, kpt_file, extract_time, image_BGR.shape[:2]))
        logger.info("================ End Extractiton ================ \n")

    logger.info('\nMean time: {:.5f}, total time: {:.5f} with {} images.'.format(np.array(total_time).mean(), np.array(total_time).sum(), len(np.array(total_time))))
    logger.info('\n{} images have no keypoints extracted.'.format(counter_fail_extraction))

if __name__ == '__main__':
    main()