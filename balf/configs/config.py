import argparse
from ..utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='motion blur feature detection')

    # Basic configuration
    parser.add_argument('--cfg_file', type=str, default='configs/gopro_train_detection.yaml')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Path for saving checkpoints')
    parser.add_argument('--exper_name', type=str)
    parser.add_argument('--fix_random_seed', type=bool, default=False,
                        help='The random seed value for PyTorch and Numpy.')
    parser.add_argument('--random_seed', type=int, default=233,
                        help='The random seed value for PyTorch and Numpy.')
    parser.add_argument('--is_debugging', type=bool, default=False,
                        help='Set variable to True if you desire to train network on a smaller dataset.')
    parser.add_argument('--resume_training', type=str, default=None,
                        help='Set saved model parameters if resume training is desired.')

    parser.add_argument('--val_data_dir', type=str, default='data/hpatches-sequences-release/',
                        help='The root path to HSequences dataset.')
    parser.add_argument('--split', type=str, default='full',
                        help='The name of the HPatches (HSequences) split. Use full, debug_view, debug_illum, view or illum.')
    parser.add_argument('--split_path', type=str, default='benchmark_test/splits.json',
                        help='The path to the split json file.')

    args = parser.parse_args()

    cfg = common_utils.get_cfg_from_yaml_file(args.cfg_file)

    return args, cfg

def parse_test_config():
    parser = argparse.ArgumentParser(description='motion blur feature matching test')

    # Basic configuration
    parser.add_argument('--cfg_file', type=str, default='balf/configs/test.yaml')
    parser.add_argument('--ckpt_file', type=str, default = 'pretrained/balf/balf.pth', 
                        help='The path to the checkpoint file to load the detector weights.')
    parser.add_argument('--ckpt_descriptor_file', type=str, default = 'pretrained/hardnet/HardNet++.pth', 
                        help='The path to the checkpoint file to load the descriptor weights.')
    parser.add_argument('--border_size', type=int, default=15,
                        help='The number of pixels to remove from the borders to compute the repeatability.')
    parser.add_argument('--nms_size', type=int, default=15,
                        help='The NMS size for computing the validation repeatability.')
    parser.add_argument('--num_features', type=int, default=2048,
                        help='The number of desired features to extract.')
    parser.add_argument('--s_mult', type=int, default=60,
                        help='The scale of laf.')
    parser.add_argument('--order_coord', type=str, default='xysr',
                        help='The coordinate order that follows the extracted points. Use yxsr or xysr.')
    parser.add_argument('--heatmap_confidence_threshold', type=float, default=0.001,
                        help='Keypoints confidence threshold.')
    parser.add_argument('--sub_pixel', type=bool, default=True,
                        help='Extract subpixel detection.')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Subpixel patch size.')

    args = parser.parse_args()

    cfg = common_utils.get_cfg_from_yaml_file(args.cfg_file)

    return args, cfg