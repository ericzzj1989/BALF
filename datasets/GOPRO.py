import os
import logging
import numpy as np
from pathlib import Path

from datasets import COCO


class GOPRO(COCO.COCO):
    def get_sequence_names(self):
        sequences_file = (
            Path(self.dataset_cfg['sequences_split_path'], 'train.txt')
            if self.task == 'train'
            else Path(self.dataset_cfg['sequences_split_path'], 'val.txt')
        )

        self.sequences_name = open(sequences_file).read() # self.sequences: GOPR0372_07_00
        print("Get {} sequences name from {}".format(len(open(sequences_file).readlines()), sequences_file))

    def get_image_paths(self):
        images_info = []

        logging.info('Get {} images from the below path:'.format(self.task))
        for r, d, f in os.walk(self.dataset_cfg['images_path']):
            if 'blur_gamma' in r or 'sharp' in r:
                if r.split('/')[-2] not in self.sequences_name:
                    continue
                logging.info(r)
                for file_name in f:
                    if file_name.endswith(".JPEG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                        images_info.append(Path(r, file_name))
        
        src_idx = np.random.permutation(len(np.asarray(images_info)))
        images_info = np.asarray(images_info)[src_idx]
        return images_info


    def init_dataset(self):
        self.get_sequence_names()
        self.images_paths = self.get_image_paths()
        print("Get {} images from {}".format(len(self.images_paths), self.dataset_cfg['images_path']))
        return len(self.images_paths[:200]), self.images_paths[:200]