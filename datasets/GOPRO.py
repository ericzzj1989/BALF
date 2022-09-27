import os
import numpy as np
from pathlib import Path

from datasets import COCO
from utils import common_utils


class GOPRO(COCO.COCO):
    def __init__(self, data_cfg, dataset_name, task, is_debugging=False):
        self.config = data_cfg
        self.dataset_cfg = data_cfg[dataset_name]
        self.dataset_name = dataset_name
        self.task = task
        self.is_debugging=is_debugging
        self.data = []

        self.create_save_path()
        self.get_sequence_names()
        synth_data_exist = self.check_synth_data_exist()
        self.get_synthetic_pairs(synth_data_exist)

    def get_sequence_names(self):
        sequences_file = (
            Path(self.dataset_cfg['sequences_split_path'], 'train.txt')
            if self.task == 'train'
            else Path(self.dataset_cfg['sequences_split_path'], 'val.txt')
        )

        self.sequences_name = open(sequences_file).read() # self.sequences: GOPR0372_07_00
        print("Get {} sequences name from {}".format(len(open(sequences_file).readlines()), sequences_file))

    def get_synthetic_pairs(self, synth_data_exist):
        if not synth_data_exist:
            common_utils.check_directory(self.save_path)
            self.images_info = self.get_image_paths()

            print("Get {} images from {}".format(len(self.images_info), self.dataset_cfg['images_path']))

            self.generate_synthetic_pairs()

        else:
            self.load_synthetic_pairs()

    def get_image_paths(self):
        images_info = []

        for r, d, f in os.walk(self.dataset_cfg['images_path']):
            if 'blur_gamma' in r or 'sharp' in r:
                if r.split('/')[-2] not in self.sequences_name:
                    continue
                for file_name in f:
                    if file_name.endswith(".JPEG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                        images_info.append(Path(r, file_name))
        
        src_idx = np.random.permutation(len(np.asarray(images_info)))
        images_info = np.asarray(images_info)[src_idx]
        return images_info