import random
import yaml
import numpy as np
from pathlib import Path
import importlib
import torch


def get_cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)

    return config

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_directory(file_path):
    file_path = Path(file_path)
    file_path.mkdir(parents=True, exist_ok=True)

def get_writer_path(exper_name, start_time):
    output_dir = Path('runs')
    return str(output_dir / exper_name / start_time)


def get_module(path, name):
    if path == '':
        mod = importlib.import_module(name)
    else:
        mod = importlib.import_module('{}.{}'.format(path, name))
    return getattr(mod, name)


def remove_borders(images, borders=3):
    ## input [B,C,H,W]
    shape = images.shape

    if len(shape) == 4:
        for batch_id in range(shape[0]):
            images[batch_id, :, 0:borders, :] = 0
            images[batch_id, :, :, 0:borders] = 0
            images[batch_id, :, shape[2] - borders:shape[2], :] = 0
            images[batch_id, :, :, shape[3] - borders:shape[3]] = 0
    # elif len(shape) == 3:
    #     ## C, H, W case
    #     images[:, 0:borders, :] = 0
    #     images[:, :, 0:borders] = 0
    #     images[:, shape[1] - borders:shape[1], :] = 0
    #     images[:, :, shape[2] - borders:shape[2]] = 0
    # else:
    #     images[0:borders, :] = 0
    #     images[:, 0:borders] = 0
    #     images[shape[0] - borders:shape[0], :] = 0
    #     images[:, shape[1] - borders:shape[1]] = 0
    else:
        print("Not implemented")
        exit()

    return images