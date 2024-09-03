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