from torch.utils.data import Sampler
import torch.distributed as dist
from module.dataset import DevDataset

class LasagnaSampler(Sampler):
    def __init__(self, dataset):
        assert type(dataset) == DevDataset
        self.dataset = dataset
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.part_size = dataset.part_size
        
        self.batch_size = self.world_size // self.part_size
        self.len_data = len(self.dataset)
        self.batch_count = self.len_data // self.batch_size
        self.group_rank = self.rank // self.part_size

    def __iter__(self):
        indices = list(range(self.len_data))
        lasagna_indices = indices[self.group_rank: :self.batch_size][:self.batch_count]
        return iter(lasagna_indices)

    def __len__(self):
        return self.batch_count
