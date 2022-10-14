import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2

import torch
import torch.nn.functional as F

from skimage.transform import pyramid_gaussian

from configs import config_hpatches
from utils import common_utils, train_utils
from utils.logger import logger
from model import get_model
from datasets import dataset_utils
from benchmark_test import geometry_tools, repeatability_tools


def generate_score_map(image_RGB_norm, model, device, config, is_debugging):
    height_RGB_norm, width_RGB_norm = image_RGB_norm.shape[0], image_RGB_norm.shape[1]
    
    image_even = dataset_utils.make_shape_even(image_RGB_norm)
    height_even, width_even = image_even.shape[0], image_even.shape[1]
    
    image_pad = dataset_utils.mod_padding_symmetric(image_even, factor=64)
    

    image_pad_tensor = torch.tensor(image_pad, dtype=torch.float32)
    image_pad_tensor = image_pad_tensor.permute(2, 0, 1)
    image_pad_batch = image_pad_tensor.unsqueeze(0)


    output_pad_batch = model(image_pad_batch.to(device))


    score_map_pad_batch = F.relu(train_utils.depth_to_space_without_softmax(output_pad_batch, config['cell_size']))
    score_map_pad_np = score_map_pad_batch[0, 0, :, :].cpu().detach().numpy()

    # unpad images to get the original resolution
    new_height, new_width = score_map_pad_np.shape[0], score_map_pad_np.shape[1]
    h_start = new_height // 2 - height_even // 2
    h_end = h_start + height_RGB_norm
    w_start = new_width // 2 - width_even // 2
    w_end = w_start + width_RGB_norm
    score_map = score_map_pad_np[h_start:h_end, w_start:w_end]

    if is_debugging:
        dataset_utils.debug_test_multiscale_results(
            image_RGB_norm, image_even, image_pad,
            score_map_pad_np, score_map
        )

    return score_map


def extract_features(image, model, device, levels, point_level, config, args, is_debugging=False):
    pyramid = pyramid_gaussian(image, max_layer=args.pyramid_levels, downscale=args.scale_factor_levels)

    score_maps = {}
    for (j, resized) in enumerate(pyramid):
        im = resized.reshape(resized.shape[0], resized.shape[1], 3)

        im_scores = generate_score_map(im, model, device, config, is_debugging)
        im_scores = geometry_tools.remove_borders(im_scores, borders=config['border_size'])

        score_maps['map_' + str(j + 1 + args.upsampled_levels)] = im_scores[:, :]

    if args.upsampled_levels:
        for j in range(args.upsampled_levels):
            factor = args.scale_factor_levels ** (args.upsampled_levels - j)
            up_image = cv2.resize(image, (0, 0), fx=factor, fy=factor)

            im = np.reshape(up_image, (up_image.shape[0], up_image.shape[1], 3))

            im_scores = generate_score_map(im, model, device, config, is_debugging)
            im_scores = geometry_tools.remove_borders(im_scores, borders=config['border_size'])

            score_maps['map_' + str(j + 1)] = im_scores[:, :]

    im_pts = []
    for idx_level in range(levels):
        scale_value = (args.scale_factor_levels ** (idx_level - args.upsampled_levels))
        scale_factor = 1. / scale_value

        h_scale = np.asarray([[scale_factor, 0., 0.], [0., scale_factor, 0.], [0., 0., 1.]])
        h_scale_inv = np.linalg.inv(h_scale)
        h_scale_inv = h_scale_inv / h_scale_inv[2, 2]

        num_points_level = point_level[idx_level]
        if idx_level > 0:
            res_points = int(np.asarray([point_level[a] for a in range(0, idx_level + 1)]).sum() - len(im_pts))
            num_points_level = res_points
        
        im_scores = repeatability_tools.apply_nms(score_maps['map_' + str(idx_level + 1)], config['nms_size'])

        im_pts_tmp = geometry_tools.get_point_coordinates(im_scores, num_points=num_points_level, order_coord='xysr')
        
        im_pts_tmp = geometry_tools.apply_homography_to_points(im_pts_tmp, h_scale_inv)

        if not idx_level:
            im_pts = im_pts_tmp
        else:
            im_pts = np.concatenate((im_pts, im_pts_tmp), axis=0)

    
    pts_sorted = im_pts[(-1 * im_pts[:, 3]).argsort()]
    pts_output = pts_sorted[:config['num_points']]

    return pts_output # pts_output.shape: N*4



def main():
    args, cfg = config_hpatches.parse_multiscale_config()

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

    ## points level define
    point_level = []
    tmp = 0.0
    factor_points = (args.scale_factor_levels ** 2)
    levels = args.pyramid_levels + args.upsampled_levels + 1
    for idx_level in range(levels):
        tmp += factor_points ** (-1 * (idx_level - args.upsampled_levels))
        point_level.append(cfg['num_points'] * factor_points ** (-1 * (idx_level - args.upsampled_levels)))


    point_level = np.asarray(list(map(lambda x: int(x/tmp), point_level)))

    list_file = open(args.hsequences_list_file, 'r')
    images_list = sorted(list_file.readlines())
    iterate = tqdm(images_list, total=len(images_list), desc="Motion Blur Detection")

    for path_to_image in iterate:
        image_path = path_to_image.rstrip('\n')
        iterate.set_description("Current {}".format(image_path))

        if not Path(image_path).exists():
            print('[ERROR]: File {0} not found!'.format(image_path))
            return

        if 'blur' in image_path:
            image_path = image_path.replace('/result', '')

        image_BGR = dataset_utils.read_bgr_image(str(image_path))
        image_RGB = dataset_utils.bgr_to_rgb(image_BGR)
        image_RGB_norm = image_RGB / 255.0

        with torch.no_grad():
            image_pts = extract_features(image_RGB_norm, model, device, levels, point_level, cfg, args)


        save_pts_dir = Path(detection_output_dir, image_path)
        common_utils.create_result_dir(str(save_pts_dir))
        
        kpt_file = Path(str(save_pts_dir)+'.kpt')

        assert image_path.split('/')[-1] == str(kpt_file).split('/')[-1][:-4]

        np.savez(kpt_file, kpts=image_pts)

        logger.info('{} kpts saved in {}'.format(image_pts.shape, kpt_file))


if __name__ == '__main__':
    main()