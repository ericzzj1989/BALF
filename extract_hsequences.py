import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F

from configs import config_hpatches
from utils import common_utils, train_utils
from utils.logger import logger
from model import get_model
from datasets import dataset_utils
from benchmark_test import geometry_tools, repeatability_tools


def extract_features(image_RGB_norm, model, device, config, is_debugging=False):
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


    score_map_remove_border = geometry_tools.remove_borders(score_map, borders=config['border_size'])
    score_map_nms = repeatability_tools.apply_nms(score_map_remove_border, config['nms_size'])


    if is_debugging:
        dataset_utils.debug_test_results(
            image_RGB_norm, image_even, image_pad, score_map_pad_np,
            score_map, score_map_remove_border, score_map_nms
        )


    pts = geometry_tools.get_point_coordinates(score_map_nms, num_points=config['num_points'], order_coord='xysr')

    pts_sorted = pts[(-1 * pts[:, 3]).argsort()]
    pts_output = pts_sorted[:config['num_points']]

    return pts_output # pts_output.shape: N*4



def main():
    args, cfg = config_hpatches.parse_config()

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
            image_pts = extract_features(image_RGB_norm, model, device, cfg)


        save_pts_dir = Path(detection_output_dir, image_path)
        common_utils.create_result_dir(str(save_pts_dir))
        
        kpt_file = Path(str(save_pts_dir)+'.kpt')

        assert image_path.split('/')[-1] == str(kpt_file).split('/')[-1][:-4]

        np.savez(kpt_file, kpts=image_pts)

        logger.info('{} kpts saved in {}'.format(image_pts.shape, kpt_file))


if __name__ == '__main__':
    main()