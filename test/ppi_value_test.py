import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
import dgl
import dgl.function as fn
from dgl.data import TUDataset
from dgl.partition import metis_partition
import torch.distributed as dist
import os, sys
import copy
from datetime import timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.layer import GCNLayer
from module.model import GCNPPI_SAGE, GCNProtein
from module.dataset import DevDataset, custom_collate_fn
from module.sampler import LasagnaSampler
from helper.all_to_all import all_to_all
from helper.utils import average_loss

# 初始化分布式环境
def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29515'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=size, init_method='env://', timeout=timedelta(minutes=1.5))
    fn(rank, size)

def run(rank, size):
    print(f"Rank {rank}: 进入run函数")
    # gcn_layer = GCNLayer(in_feats=3, out_feats=3, num_parts=num_parts)
    part_size = 2
    torch.manual_seed(43)
    # model = GCNProtein(in_feats=1, out_feats=1, part_size=part_size)
    model = GCNPPI_SAGE(in_feats=50, out_feats=121, part_size=part_size).cuda()
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 weight_decay=0)
    train_dataset = DevDataset("ppi", part_size=part_size, mode='train', process_data=False)
    # train_dataset = DevDataset("proteins", part_size=part_size)
    train_sampler = LasagnaSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=False, collate_fn=custom_collate_fn)
    
    for epoch in range(5000):
        model.train()
        total_loss = 0
        for g_strt, feat, tag in train_loader:
            feat.requires_grad_(True)
            device = torch.device('cuda')
            g_strt = g_strt.to(device)
            feat = feat.to(device)
            tag = tag.to(device)
            output = model.forward(g_strt, feat)
            # print("Rank", rank, '\n',
            #     "节点的全局序号:", g_strt.lasagna_data['_ID'].tolist(), '\n',
            #     "输出特征：", output, '\n',
            #     "节点 target:", tag,
            # )
            loss = criterion(output, tag)
            average_loss(loss, g_strt.lasagna_data['n_node'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"Rank {rank} 训练后的参数： {model.gcnLayer1.linear.weight} {model.gcnLayer1.linear.bias} {model.gcnLayer2.linear.weight} {model.gcnLayer2.linear.bias}")
            # print(f"Rank {rank} 训练后的参数： {model.linear1.weight} {model.linear1.bias}")
            # print(f"Rank {rank} 训练后feat的梯度： {feat.grad}")
            total_loss += loss.item()
            # print(f'Rank {rank} Epoch {epoch + 1}, Loss: {loss:.4f}')
        total_loss = torch.tensor(total_loss, dtype=torch.float, device='cuda')
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            print(f'Rank {rank} Epoch {epoch + 1}, Total Loss: {total_loss:.4f}')


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    size = 8
    torch.multiprocessing.set_start_method('spawn')
    processes = []
    for rank in range(size):
        p = torch.multiprocessing.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

