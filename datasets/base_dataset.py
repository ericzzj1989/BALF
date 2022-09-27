import logging

import torch
import torch.utils.data as torch_data


class base_dataset(torch_data.Dataset):
    def __init__(self, data, task='train'):
        self.data =data

        # Restrict the number of training and validation sample (9000 : 3000 = train : val)
        if task == 'train':
            if len(self.data) > 9000:
                self.data = self.data[:9000]
        elif task == 'val':
            if len(self.data) > 3000:
                self.data = self.data[:3000]

        logging.info('Task : {} the number of sample : {}'.format(task, len(self.data)))        

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        im_src_patch, im_dst_patch, heatmap_src_patch, heatmap_dst_patch, homography_src_2_dst, homography_dst_2_src = self.data[idx]
        
        im_src_patch = torch.tensor(im_src_patch, dtype=torch.float32)
        im_dst_patch = torch.tensor(im_dst_patch, dtype=torch.float32)
        heatmap_src_patch = torch.tensor(heatmap_src_patch, dtype=torch.float32)
        heatmap_dst_patch = torch.tensor(heatmap_dst_patch, dtype=torch.float32)
        homography_src_2_dst = torch.tensor(homography_src_2_dst, dtype=torch.float32)
        homography_dst_2_src = torch.tensor(homography_dst_2_src, dtype=torch.float32)

        return im_src_patch.permute(2, 0, 1), im_dst_patch.permute(2, 0, 1), heatmap_src_patch.unsqueeze(0), heatmap_dst_patch.unsqueeze(0), homography_src_2_dst, homography_dst_2_src