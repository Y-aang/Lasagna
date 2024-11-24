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
from module.model import GCNProtein
from module.dataset import DevDataset, custom_collate_fn
from module.sampler import LasagnaSampler
from helper.all_to_all import all_to_all
from helper.utils import register_hook_for_model

# 初始化分布式环境
def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29515'
    dist.init_process_group(backend, rank=rank, world_size=size, init_method='env://', timeout=timedelta(minutes=1.5))
    fn(rank, size)
# from buffer import LocalFeatureStore
# feature_store = LocalFeatureStore()
# 第4步：定义跨多分区消息传递的 GCN 层

# 第5步：运行带有跨分区消息传递的GCN
def run(rank, size):
    print(f"Rank {rank}: 进入run函数")
    # processing data
    # for data(graph structure & graph feature) in dataset:
    
    # TODO: gain data from path
    # gcn_layer = GCNLayer(in_feats=3, out_feats=3, num_parts=num_parts)
    gcn_module = GCNProtein(in_feats=1, out_feats=1, num_parts=4)
    criterion = nn.L1Loss(reduction='sum')
    optimizer = optim.SGD(gcn_module.parameters(), lr=0.001)
    train_dataset = DevDataset("proteins", datasetPath=None)
    train_sampler = LasagnaSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=train_sampler, shuffle=False, collate_fn=custom_collate_fn)
    
    for epoch in range(10):
        gcn_module.train()
        total_loss = 0
        for part, send_map, recv_map, g_structure in train_loader:
            part.ndata['h'].requires_grad_(True)
            output = gcn_module.forward(g_structure, part.ndata['h'], part.ndata['norm'], send_map, recv_map, rank, size)
            # print("Rank", rank, '\n',
            #     "节点的全局序号:", part.ndata['_ID'].tolist(), '\n',
            #     "输出特征：", output, '\n',
            #     "节点 target:", part.ndata['tag'],
            # )
            loss = criterion(output, part.ndata['tag'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(f"Rank {rank} 训练后的参数： {gcn_module.gcnLayer1.linear.weight} {gcn_module.gcnLayer1.linear.bias}")
            # print(f"Rank {rank} 训练后feat的梯度： {part.ndata['h'].grad}")
            total_loss += loss.item()
        print(f'Rank {rank} Epoch {epoch + 1}, Loss: {total_loss:.4f}')


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    size = 4  # 使用4个进程 (每个GPU一个子图)
    torch.multiprocessing.set_start_method('spawn')
    processes = []

    for rank in range(size):
        p = torch.multiprocessing.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

