import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from dataSet import LasagnaDataset

def run(rank, device):
    print(f"Process {rank} running on device {device}")
    torch.cuda.set_device(device)
    
    datasetPath = "/data/yangshen/template/BNS-GCN/dataset"
    dataset = LasagnaDataset("reddit", datasetPath)
    

def init_process(rank, n_gpus):
    device = f'cuda:{rank % n_gpus}'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    run(rank, device)
    dist.destroy_process_group()

if __name__ == "__main__":
    n_gpus = 4
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '18118'
    
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for rank in range(n_gpus):
        p = mp.Process(target=init_process, args=(rank, n_gpus))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
