import torch.utils.data as torch_data


class base_dataset(torch_data.Dataset):
    def init_dataset(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.len

    def __init__(self, data_cfg, dataset_name, task, is_debugging=False):
        # Update config
        super().__init__()
        self.len = 0
        self.config = data_cfg
        self.dataset_cfg = data_cfg[dataset_name]
        self.dataset_name = dataset_name
        self.task = task
        self.is_debugging = is_debugging
        self.len, self.train_files = self.init_dataset()