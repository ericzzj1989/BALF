import numpy as np
from pathlib import Path
import json

from datasets import dataset_utils

class Resize_HSequences(object):
    
    def __init__(self, dataset_path, split, split_path, args):

        self.dataset_path = dataset_path
        self.split = split
        self.args = args

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

        images_dst_BGR = []
        homographies = []

        sequence_path = Path(self.dataset_path, self.sequences[folder_id])
        image_src_path = str(sequence_path) + '/1.ppm'

        im_src_BGR = dataset_utils.read_bgr_image(image_src_path)

        for i in range(5):

            image_dst_path = str(sequence_path) + '/' + str(i+2) + '.ppm'
            assert image_src_path.split('/')[-2] == image_dst_path.split('/')[-2]

            # image_dst_path = str(sequence_path) + '/result/' + str(i+2) + '.ppm'
            # assert image_src_path.split('/')[-2] == image_dst_path.split('/')[-3]

            print('\ndst image path: ', image_dst_path)

            im_dst_BGR = dataset_utils.read_bgr_image(image_dst_path)

            homography_path = str(sequence_path) + '/H_1_' + str(i+2)
            homography = np.loadtxt(homography_path)

            if self.args.resize_image:
                src_shape = im_src_BGR.shape[:2]
                dst_shape = im_dst_BGR.shape[:2]
                homography = {'homography': homography, 'shape': np.array(src_shape), 'warped_shape': np.array(dst_shape)}
                homography = dataset_utils.adapt_homography_to_preprocessing(homography, self.args)

                im_dst_BGR = dataset_utils.ratio_preserving_resize(im_dst_BGR, self.args.resize_shape)

            images_dst_BGR.append(im_dst_BGR)
            homographies.append(homography)

        if self.args.resize_image:
            im_src_BGR = dataset_utils.ratio_preserving_resize(im_src_BGR, self.args.resize_shape)

        images_dst_BGR = np.asarray(images_dst_BGR)
        homographies = np.asarray(homographies)

        print(self.sequences[folder_id])

        return {'im_src_BGR': im_src_BGR, 'images_dst_BGR': images_dst_BGR,
                'homographies': homographies, 'sequence_name': self.sequences[folder_id]}



    def get_hsequences(self):

        for idx_sequence in range(len(self.sequences)):

            yield self.get_sequence(idx_sequence)