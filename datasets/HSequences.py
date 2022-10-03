import numpy as np
from pathlib import Path
import json

from datasets import dataset_utils

class HSequences(object):
    
    def __init__(self, dataset_path, split, split_path):

        self.dataset_path = dataset_path
        self.split = split

        self.splits = json.load(open(split_path))
        self.sequences = self.splits[self.split]['test']

        self.count = 0

    def read_homography(self, h_name):

        h = np.zeros((3, 3))
        h_file = open(h_name, 'r')

        # Prepare Homography
        for j in range(3):
            line = h_file.readline()
            line = str.split(line);
            for i in range(3):
                h[j, i] = float(line[i])

        inv_h = np.linalg.inv(h)
        inv_h = inv_h / inv_h[2, 2]

        return h, inv_h

    def get_sequence_data(self, folder_id):

        images_dst_RGB_norm = []
        h_src_2_dst = []
        h_dst_2_src = []

        sequence_path = Path(self.dataset_path, self.sequences[folder_id])
        image_src_path = str(sequence_path) + '/1.ppm'

        im_src_BGR = dataset_utils.read_bgr_image(image_src_path)
        im_src_RGB = dataset_utils.bgr_to_rgb(im_src_BGR)
        im_src_RGB_norm = im_src_RGB / 255.0

        for i in range(5):

            image_dst_path = str(sequence_path) + '/' + str(i+2) + '.ppm'

            assert image_src_path.split('/')[-2] == image_dst_path.split('/')[-2]

            im_dst_BGR = dataset_utils.read_bgr_image(image_dst_path)
            im_dst_RGB = dataset_utils.bgr_to_rgb(im_dst_BGR)
            im_dst_RGB_norm = im_dst_RGB / 255.0

            images_dst_RGB_norm.append(im_dst_RGB_norm)

            homography_path = str(sequence_path) + '/H_1_' + str(i+2)
            src_2_dst, dst_2_src = self.read_homography(homography_path)
            h_src_2_dst.append(src_2_dst)
            h_dst_2_src.append(dst_2_src)

        images_dst_RGB_norm = np.asarray(images_dst_RGB_norm)
        h_src_2_dst = np.asarray(h_src_2_dst)
        h_dst_2_src = np.asarray(h_dst_2_src)

        print(self.sequences[folder_id])

        return {'im_src_RGB_norm': im_src_RGB_norm, 'images_dst_RGB_norm': images_dst_RGB_norm,
                'h_src_2_dst': h_src_2_dst, 'h_dst_2_src': h_dst_2_src,
                'sequence_name': self.sequences[folder_id]}



    def get_hsequences(self):

        for idx_sequence in range(len(self.sequences)):

            yield self.get_sequence(idx_sequence)