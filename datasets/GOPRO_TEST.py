import numpy as np
from pathlib import Path
import json

from datasets import dataset_utils

class GOPRO_test(object):
    
    def __init__(self, dataset_path, split, split_path):

        self.dataset_path = dataset_path
        self.split = split

        self.splits = json.load(open(split_path))
        self.sequences = self.splits['test']

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

        images_dst_BGR = []
        h_src_2_dst = []
        h_dst_2_src = []

        sequence_path = Path(self.dataset_path, self.sequences[folder_id])

        if self.split == 'src_blur_dst_sharp':
            image_src_path = str(sequence_path) + '/blur_gamma/1.png'
        elif self.split == 'src_sharp_dst_blur':
            image_src_path = str(sequence_path) + '/sharp/1.png'
        elif self.split == 'src_blur_dst_blur':
            image_src_path = str(sequence_path) + '/blur_gamma/1.png'
        elif self.split == 'src_blur_dst_blur_diff':
            image_src_path = str(sequence_path) + '/blur_gamma/1.png'

        print('\n*****src image path: ', image_src_path)

        im_src_BGR = dataset_utils.read_bgr_image(image_src_path)

        for i in range(6):

            if self.split == 'src_blur_dst_sharp':
                image_dst_path = str(sequence_path) + '/sharp/' + str(i+1) + '.png'
            elif self.split == 'src_sharp_dst_blur':
                image_dst_path = str(sequence_path) + '/blur_gamma/' + str(i+1) + '.png'
            elif self.split == 'src_blur_dst_blur':
                image_dst_path = str(sequence_path) + '/blur_gamma/' + str(i+1) + '.png'
            elif self.split == 'src_blur_dst_blur_diff':
                image_dst_path = str(sequence_path) + '/blur_diff/' + str(i+1) + '.png'

            assert image_src_path.split('/')[-3] == image_dst_path.split('/')[-3]

            print('*****dst image path: ', image_dst_path)

            im_dst_BGR = dataset_utils.read_bgr_image(image_dst_path)

            images_dst_BGR.append(im_dst_BGR)

            homography_path = str(sequence_path) + '/H_1_' + str(i+1)
            src_2_dst, dst_2_src = self.read_homography(homography_path)
            h_src_2_dst.append(src_2_dst)
            h_dst_2_src.append(dst_2_src)

        images_dst_BGR = np.asarray(images_dst_BGR)
        h_src_2_dst = np.asarray(h_src_2_dst)
        h_dst_2_src = np.asarray(h_dst_2_src)

        print(self.sequences[folder_id])

        return {'im_src_BGR': im_src_BGR, 'images_dst_BGR': images_dst_BGR,
                'h_src_2_dst': h_src_2_dst, 'h_dst_2_src': h_dst_2_src,
                'sequence_name': self.sequences[folder_id]}



    def get_hsequences(self):

        for idx_sequence in range(len(self.sequences)):

            yield self.get_sequence(idx_sequence)