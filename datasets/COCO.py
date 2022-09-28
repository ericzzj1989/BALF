import os
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

from utils import common_utils
from datasets import dataset_utils

class COCO(object):
    def __init__(self, data_cfg, dataset_name, task, is_debugging=False):
        self.config = data_cfg
        self.dataset_cfg = data_cfg[dataset_name]
        self.dataset_name = dataset_name
        self.task = task
        self.is_debugging=is_debugging
        self.data = []

        self.create_save_path()

        synth_data_exist = self.check_synth_data_exist()
        self.get_synthetic_pairs(synth_data_exist)

        # self.get_image_paths()

        # self.creat_synthetic_pairs()

    def get_synthetic_pairs(self, synth_data_exist):
        if not synth_data_exist:
            common_utils.check_directory(self.save_path)
            self.images_info = self.get_image_paths()

            print("Get {} images from {}".format(len(self.images_info), self.dataset_cfg['images_path']))

            self.generate_synthetic_pairs()

        else:
            self.load_synthetic_pairs()

        print()

    def get_image_paths(self):
        images_info = []
        for r, d, f in os.walk(Path(self.dataset_cfg['images_path'])):
            if self.task not in r:
                continue
            for file_name in f:
                if file_name.endswith(".JPEG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                    images_info.append(Path(r, file_name))

        src_idx = np.random.permutation(len(np.asarray(images_info)))
        images_info = np.asarray(images_info)[src_idx]
        print('images_info:\n',images_info)

        return images_info



    def create_save_path(self):
        if self.is_debugging:
            self.save_path = Path(self.config['synth_dir'], self.dataset_name, self.task + '_dataset_debug')
        else:
            self.save_path = Path(self.config['synth_dir'], self.dataset_name, self.task + '_dataset')

    def check_synth_data_exist(self):
        return self.save_path.is_dir()

    def get_data(self):
        return self.data

    def load_synthetic_pairs(self):
        print('\n================= Loading Synthetic Pairs ================')

        # save_path: data/synth/GOPRO/train_dataset
        save_path = self.save_path

        counter = 0

        for r, d, f in tqdm(os.walk(save_path)):
            for file_name in f:
                if file_name.endswith(".JPEG,npz") or file_name.endswith(".jpg.npz") or file_name.endswith(".png.npz"):
                    synthetic_pair_path = Path(r, file_name)
                    im_src_patch = np.load(synthetic_pair_path, allow_pickle=True)['im_src_patch']
                    im_dst_patch = np.load(synthetic_pair_path, allow_pickle=True)['im_dst_patch']
                    heatmap_src_patch = np.load(synthetic_pair_path, allow_pickle=True)['heatmap_src_patch']
                    heatmap_dst_patch = np.load(synthetic_pair_path, allow_pickle=True)['heatmap_dst_patch']
                    homography_src_2_dst = np.load(synthetic_pair_path, allow_pickle=True)['homography_src_2_dst']
                    homography_dst_2_src = np.load(synthetic_pair_path, allow_pickle=True)['homography_dst_2_src']
                    
                    self.data.append([im_src_patch, im_dst_patch,
                                      heatmap_src_patch, heatmap_dst_patch,
                                      homography_src_2_dst, homography_dst_2_src])
                    
                    counter += 1

        print('Load {} synthetic pairs from {}.'.format(counter, save_path))
        print('================= Finish Loading Synthetic Pairs ================\n')

    def generate_synthetic_pairs(self):
        if self.task == 'train':
            patch_size = self.config['patch_size']
            self.counter = 0
        else:
            patch_size = 2 * self.config['patch_size']
            self.counter = 0

        counter_patches = 0

        print('\n================= Generating Synthetic Pairs ================')
        print('Create pairs with {}x{}'.format(patch_size, patch_size))

        save_path = self.save_path

        counter = 0
        for path_image_idx in tqdm(range(len(self.images_info))):
            # image_path example: data/GOPRO_Large/train/GOPR0372_07_01/sharp/0000606.png
            # data/COCO/train2014/COCO_train2014_00004197.jpg
            image_path = self.images_info[(self.counter+path_image_idx) % len(self.images_info)]
            
            correct_patch = False
            counter = -1
            while counter < 10:
                counter += 1
                incorrect_h = True

                while incorrect_h:

                    src_BGR = dataset_utils.read_bgr_image(str(image_path))
                    src_RGB = dataset_utils.bgr_to_rgb(src_BGR)

                    source_shape = src_RGB.shape
                    h = dataset_utils.generate_homography(source_shape, self.config['homographic'])

                    inv_h = np.linalg.inv(h)
                    inv_h = inv_h / inv_h[2, 2]

                    dst_RGB = dataset_utils.bgr_photometric_to_rgb(src_BGR)
                    dst_RGB = dataset_utils.apply_homography_to_source_image(dst_RGB, inv_h)

                    if dst_RGB.max() > 0.0:
                        incorrect_h = False

                
                if self.dataset_name == 'COCO':
                    label_path = Path(self.dataset_cfg['labels_path'], "{}.npz".format(image_path.stem))
                elif self.dataset_name == 'GOPRO':
                    label_path = Path(self.dataset_cfg['labels_path'], image_path.parts[-3], "{}.npz".format(image_path.stem))

                src_label = np.load(label_path, allow_pickle=True)['pts']
                src_label_k_best = dataset_utils.select_k_best(src_label, 0)
                src_heatmap = dataset_utils.labels_to_heatmap(src_label_k_best, source_shape)

                inv_h_tensor = torch.tensor(inv_h, dtype=torch.float32)
                dst_heatmap_tensor = dataset_utils.apply_homography_to_source_labels_torch(src_label_k_best, source_shape, inv_h_tensor, bilinear=True)
                dst_heatmap = dst_heatmap_tensor.squeeze().numpy()

                src_RGB_norm = src_RGB / 255.0
                dst_RGB_norm = dst_RGB / 255.0

                point_src = dataset_utils.get_window_point(source_shape, patch_size)

                im_src_patch = src_RGB_norm[int(point_src[0] - patch_size / 2): int(point_src[0] + patch_size / 2),
                                            int(point_src[1] - patch_size / 2): int(point_src[1] + patch_size / 2),
                                            :]

                point_dst = inv_h.dot([point_src[1], point_src[0], 1.0])
                point_dst = [point_dst[1] / point_dst[2], point_dst[0] / point_dst[2]]

                if (point_dst[0] - patch_size / 2) < 0 or (point_dst[1] - patch_size / 2) < 0:
                    continue
                if (point_dst[0] + patch_size / 2) > source_shape[0] or (point_dst[1] + patch_size / 2) > source_shape[1]:
                    continue

                h_src_translation = np.asanyarray([[1., 0., -(int(point_src[1]) - patch_size / 2)],
                                                   [0., 1., -(int(point_src[0]) - patch_size / 2)],
                                                   [0., 0., 1.]])
                h_dst_translation = np.asanyarray([[1., 0., int(point_dst[1] - patch_size / 2)],
                                                   [0., 1., int(point_dst[0] - patch_size / 2)],
                                                   [0., 0., 1.]])

                im_dst_patch = dst_RGB_norm[int(point_dst[0] - patch_size / 2): int(point_dst[0] + patch_size / 2),
                                            int(point_dst[1] - patch_size / 2): int(point_dst[1] + patch_size / 2),
                                            :]

                heatmap_src_patch = src_heatmap[int(point_src[0] - patch_size / 2): int(point_src[0] + patch_size / 2),
                                                int(point_src[1] - patch_size / 2): int(point_src[1] + patch_size / 2)]
                heatmap_dst_patch = dst_heatmap[int(point_dst[0] - patch_size / 2): int(point_dst[0] + patch_size / 2),
                                                int(point_dst[1] - patch_size / 2): int(point_dst[1] + patch_size / 2)]
                
                if im_src_patch.shape[0] != patch_size or im_src_patch.shape[1] != patch_size:
                    continue

                correct_patch = True
                break

            if correct_patch:
                homography = np.dot(h_src_translation, np.dot(h, h_dst_translation))

                homography_dst_2_src = homography.astype('float32')
                homography_dst_2_src = homography_dst_2_src/homography_dst_2_src[2,2]

                homography_src_2_dst = np.linalg.inv(homography)
                homography_src_2_dst = homography_src_2_dst.astype('float32')
                homography_src_2_dst = homography_src_2_dst/homography_src_2_dst[2,2]

                if self.dataset_name == 'COCO':
                    synthetic_pairs = {}
                    synthetic_pairs.update({'im_src_patch': im_src_patch, 'im_dst_patch': im_dst_patch,
                                            'heatmap_src_patch': heatmap_src_patch, 'heatmap_dst_patch': heatmap_dst_patch,
                                            'homography_src_2_dst': homography_src_2_dst, 'homography_dst_2_src': homography_dst_2_src})
                                            
                    np.savez_compressed(Path(save_path, "{}.npz".format(image_path.name)), **synthetic_pairs)
                    
                elif self.dataset_name == 'GOPRO':
                    str_image_path = str(image_path)

                    synthetic_pairs = {}
                    synthetic_pairs.update({'im_src_patch': im_src_patch, 'im_dst_patch': im_dst_patch,
                                            'heatmap_src_patch': heatmap_src_patch, 'heatmap_dst_patch': heatmap_dst_patch,
                                            'homography_src_2_dst': homography_src_2_dst, 'homography_dst_2_src': homography_dst_2_src})

                    save_pair_path = Path(save_path, str_image_path.split('/')[-3], str_image_path.split('/')[-2])
                    common_utils.check_directory(save_pair_path)
                    file_name = str_image_path.split('/')[-1]

                    # save to: data/GOPRO/train_dataset/GOPR0372_07_01/sharp/0000606.npz
                    npz_file = Path(save_pair_path, "{}.npz".format(file_name))
                    if npz_file.exists():
                        print('{} already exists.'.format(npz_file))
                        continue
                    np.savez_compressed(Path(save_pair_path, "{}.npz".format(file_name)), **synthetic_pairs) 


                self.data.append([im_src_patch, im_dst_patch,
                                  heatmap_src_patch, heatmap_dst_patch,
                                  homography_src_2_dst, homography_dst_2_src])

                if self.is_debugging:
                    import matplotlib.pyplot as plt
                    # print("im_src_patch shape: ", im_src_patch.shape)
                    # print("im_dst_patch shape: ", im_dst_patch.shape)

                    # print("heatmap_src_patch shape: ", heatmap_src_patch.shape)
                    # print("heatmap_dst_patch shape: ", heatmap_dst_patch.shape)

                    # print('homography_dst_2_src shape: ',homography_dst_2_src.shape)
                    # print('homography_src_2_dst shape: ',homography_src_2_dst.shape)

                    # dataset_utils.debug_synthetic_pairs_repeatability(
                    #     heatmap_src_patch,
                    #     heatmap_dst_patch,
                    #     im_src_patch,
                    #     im_dst_patch,
                    #     homography_dst_2_src)

                    
                    # fig = plt.figure()
                    # rows = 2 ; cols = 3
                    # ax1 = fig.add_subplot(rows, cols, 1)
                    # ax1.imshow(src_RGB_norm)
                    # ax1.set_title('src_RGB_norm (src image RGB norm)')
                    # ax1.axis("off")

                    # ax2 = fig.add_subplot(rows, cols, 2)
                    # ax2.imshow(im_src_patch)
                    # ax2.set_title('im_src_patch')
                    # ax2.axis("off")

                    # ax3 = fig.add_subplot(rows, cols, 3)
                    # ax3.imshow(heatmap_src_patch, cmap='gray')
                    # ax3.set_title('heatmap_src_patch')
                    # ax3.axis("off")

                    # ax4 = fig.add_subplot(rows, cols, 4)
                    # ax4.imshow(dst_RGB_norm)
                    # ax4.set_title('dst_RGB_norm (dst image correction RGB norm)')
                    # ax4.axis("off")

                    # ax5 = fig.add_subplot(rows, cols, 5)
                    # ax5.imshow(im_dst_patch)
                    # ax5.set_title('im_dst_patch')
                    # ax5.axis("off")
                    
                    # ax6 = fig.add_subplot(rows, cols, 6)
                    # ax6.imshow(heatmap_dst_patch, cmap='gray')
                    # ax6.set_title('heatmap_dst_patch')
                    # ax6.axis("off")

                    # plt.show()


                counter_patches += 1

            if self.task == 'train' and counter_patches > 9000:
                break
            elif counter_patches > 3000:
                break
            if self.task == 'train' and self.is_debugging and counter_patches > 400:
                break
            if self.task == 'val' and self.is_debugging and counter_patches > 150:
                break


        self.counter = counter_patches
        print('Generate {} synthetic pairs in {}.'.format(self.counter, save_path))
        print('================= Finish Generating Synthetic Pairs ================\n')