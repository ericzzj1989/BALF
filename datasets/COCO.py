import os
import numpy as np
from pathlib import Path
import torch

from datasets import dataset_utils
from datasets import base_dataset 

class COCO(base_dataset.base_dataset):
    def get_image_paths(self):
        valid_val_images_name = open(self.dataset_cfg['valid_val_split_file']).read()
        images_info = []
        for r, d, f in os.walk(Path(self.dataset_cfg['images_path'])):
            if self.task not in r:
                continue
            for file_name in f:
                if file_name.endswith(".JPEG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                    image_path = Path(r, file_name)
                    image = dataset_utils.read_bgr_image(str(image_path))
                    if self.task == 'train':
                        image_size = 256
                    else:
                        image_size = 512
                    if image.shape[0] < image_size or image.shape[1] < image_size:
                        continue
                    images_info.append(Path(r, file_name))

        src_idx = np.random.permutation(len(np.asarray(images_info)))
        images_info = np.asarray(images_info)[src_idx]
        print('images_info:\n',images_info)

        return images_info
    
    def init_dataset(self):
        self.images_paths = self.get_image_paths()
        print("Get {} images from {}".format(len(self.images_paths), self.dataset_cfg['images_path']))
        if self.task == 'train':
            # num = len(self.images_paths)
            num = 1000
        else:
            num = len(self.images_paths)
        return num, self.images_paths[:num]
    
    def __getitem__(self, index):
        if self.task == 'train':
            patch_size = self.dataset_cfg['patch_size']
            self.counter = 0
        else:
            patch_size = 2 * self.dataset_cfg['patch_size']
            self.counter = 0


        image_path = self.images_paths[index]

        incorrect_patch = True
        counter = -1
        while incorrect_patch:
            counter += 1
            incorrect_h = True

            while incorrect_h:

                src_BGR = dataset_utils.read_bgr_image(str(image_path))
                src_RGB = dataset_utils.bgr_to_rgb(src_BGR)

                source_shape = src_RGB.shape
                h = dataset_utils.generate_homography(source_shape, self.config['homographic'])

                inv_h = np.linalg.inv(h)
                inv_h = inv_h / inv_h[2, 2]

                if self.task == 'train':
                    dst_RGB = dataset_utils.bgr_photometric_to_rgb(src_BGR)
                else:
                    dst_RGB = src_RGB
                    
                dst_RGB = dataset_utils.apply_homography_to_source_image(dst_RGB, inv_h)

                if dst_RGB.max() > 0.0:
                    incorrect_h = False

            
            if self.dataset_name == 'COCO':
                label_path = Path(self.dataset_cfg['labels_path'], self.task, "{}.npz".format(image_path.stem))
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
            
            if im_src_patch.shape[0] == patch_size and im_src_patch.shape[1] == patch_size and\
                im_dst_patch.shape[0] == patch_size and im_dst_patch.shape[1] == patch_size and\
                    heatmap_src_patch.shape[0] == patch_size and heatmap_src_patch.shape[1] == patch_size and\
                        heatmap_dst_patch.shape[0] == patch_size and heatmap_dst_patch.shape[1] == patch_size:
                            incorrect_patch = False


        if incorrect_patch == False:
            homography = np.dot(h_src_translation, np.dot(h, h_dst_translation))

            homography_dst_2_src = homography.astype('float32')
            homography_dst_2_src = homography_dst_2_src/homography_dst_2_src[2,2]

            homography_src_2_dst = np.linalg.inv(homography)
            homography_src_2_dst = homography_src_2_dst.astype('float32')
            homography_src_2_dst = homography_src_2_dst/homography_src_2_dst[2,2]

            
            if self.is_debugging:
                import matplotlib.pyplot as plt
                print("im_src_patch shape: ", im_src_patch.shape)
                print("im_dst_patch shape: ", im_dst_patch.shape)

                print("heatmap_src_patch shape: ", heatmap_src_patch.shape)
                print("heatmap_dst_patch shape: ", heatmap_dst_patch.shape)

                print('homography_dst_2_src shape: ',homography_dst_2_src.shape)
                print('homography_src_2_dst shape: ',homography_src_2_dst.shape)

                dataset_utils.debug_synthetic_pairs_repeatability(
                    heatmap_src_patch,
                    heatmap_dst_patch,
                    im_src_patch,
                    im_dst_patch,
                    homography_dst_2_src)

                
                fig = plt.figure()
                rows = 2 ; cols = 3
                ax1 = fig.add_subplot(rows, cols, 1)
                ax1.imshow(src_RGB_norm)
                ax1.set_title('src_RGB_norm (src image RGB norm)')
                ax1.axis("off")

                ax2 = fig.add_subplot(rows, cols, 2)
                ax2.imshow(im_src_patch)
                ax2.set_title('im_src_patch')
                ax2.axis("off")

                ax3 = fig.add_subplot(rows, cols, 3)
                ax3.imshow(heatmap_src_patch, cmap='gray')
                ax3.set_title('heatmap_src_patch')
                ax3.axis("off")

                ax4 = fig.add_subplot(rows, cols, 4)
                ax4.imshow(dst_RGB_norm)
                ax4.set_title('dst_RGB_norm (dst image correction RGB norm)')
                ax4.axis("off")

                ax5 = fig.add_subplot(rows, cols, 5)
                ax5.imshow(im_dst_patch)
                ax5.set_title('im_dst_patch')
                ax5.axis("off")
                
                ax6 = fig.add_subplot(rows, cols, 6)
                ax6.imshow(heatmap_dst_patch, cmap='gray')
                ax6.set_title('heatmap_dst_patch')
                ax6.axis("off")

                plt.show()

            im_src_patch = torch.tensor(im_src_patch, dtype=torch.float32)
            im_dst_patch = torch.tensor(im_dst_patch, dtype=torch.float32)
            heatmap_src_patch = torch.tensor(heatmap_src_patch, dtype=torch.float32)
            heatmap_dst_patch = torch.tensor(heatmap_dst_patch, dtype=torch.float32)
            homography_src_2_dst = torch.tensor(homography_src_2_dst, dtype=torch.float32)
            homography_dst_2_src = torch.tensor(homography_dst_2_src, dtype=torch.float32)

            return im_src_patch.permute(2, 0, 1), im_dst_patch.permute(2, 0, 1), heatmap_src_patch.unsqueeze(0), heatmap_dst_patch.unsqueeze(0), homography_src_2_dst, homography_dst_2_src