import argparse
import yaml

def parse_config():
    parser = argparse.ArgumentParser(description='motion blur feature detection')

    # Basic configuration
    parser.add_argument('--cfg_file', type=str, default='configs/gopro_train_detection.yaml')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Path for saving checkpoints')
    parser.add_argument('--exper_name', type=str)
    parser.add_argument('--fix_random_seed', action='store_true', default=False,
                        help='The random seed value for PyTorch and Numpy.')
    parser.add_argument('--random_seed', type=int, default=233,
                        help='The random seed value for PyTorch and Numpy.')
    parser.add_argument('--is_debugging', action='store_true', default=True,
                        help='Turn on debuging mode')
    parser.add_argument('--resume_training', type=str, default=None,
                        help='Set saved model parameters if resume training is desired.')

    args = parser.parse_args()

    cfg = get_cfg_from_yaml_file(args.cfg_file)

    return args, cfg

def get_cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)

    return config