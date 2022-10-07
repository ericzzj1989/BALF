from torch.utils.data import DataLoader

from utils.common_utils import get_module


def build_dataloaders(data_cfg, task, is_debugging=False):
    dataloaders = [
        {'name': dataset_name,
         'dataloader': get_dataloader(data_cfg, dataset_name, task, is_debugging)}
         for dataset_name in data_cfg['dataset_names']
    ]

    return dataloaders

def get_dataloader(data_cfg, dataset_name, task, is_debugging=False):
    dataset = get_module('datasets', dataset_name)(
        data_cfg=data_cfg,
        dataset_name=dataset_name,
        task=task,
        is_debugging=is_debugging
    )

    if task == 'train':
        batch_size = data_cfg['train_batch_size']
        shuffle = True
        num_workers = data_cfg['train_num_workers']
    elif task == 'val':
        batch_size = data_cfg['val_batch_size']
        shuffle = False
        num_workers = data_cfg['val_num_workers']

    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
    )

    return dataloader