import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.data import TUDataset
from dgl.partition import metis_partition
import torch.distributed as dist
import os, sys
import torch.nn.init as init

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from layer import GCNLayer

# 第1步：加载TUDataset中的PROTEINS数据集
dataset = TUDataset(name='PROTEINS')
graph, label = dataset[0]
print(f"图信息：{graph}")

# 初始化节点特征
# num_nodes = graph.num_nodes()
# feat = torch.randn(num_nodes, 3)  # 假设每个节点有3维特征
# graph.ndata['h'] = feat

# edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
# u, v = zip(*edges)
# graph = dgl.graph((u, v))
# feat = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float)
# graph.ndata['h'] = feat
# num_nodes = graph.num_nodes()

num_nodes = 12
edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]  # 形成圆环
u, v = zip(*edges)
graph = dgl.graph((u, v))
feat = torch.tensor([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 2, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=torch.float)
graph.ndata['h'] = feat

# 第2步：使用 Metis 将图分成 n 部分
num_parts = 4  # 假设我们将图分成4部分
parts = metis_partition(graph, k=num_parts)

# 提取每个节点的分区编号（通过part_id）
node_part = torch.empty(num_nodes, dtype=torch.int64)

for part_id, subgraph in parts.items():
    node_part[subgraph.ndata['_ID']] = part_id  # 将子图的节点编号映射到原始图的节点

# 第3步：记录每个子图的send和recv信息
send_map = {i: {} for i in range(num_parts)}  # 记录每个子图要发送的目标子图
recv_map = {i: {} for i in range(num_parts)}  # 记录每个子图要接收的源子图

# 获取需要跨分区传递的边 (跨分区的节点)
for u, v in zip(graph.edges()[0], graph.edges()[1]):
    part_u = node_part[u].item()  # u的分区
    part_v = node_part[v].item()  # v的分区
    if part_u != part_v:
        # u 需要发送特征给 v 的子图
        if part_v not in send_map[part_u]:
            send_map[part_u][part_v] = []
        send_map[part_u][part_v].append(u.item())       #38 ...

        # v 需要从 u 的子图接收特征
        if part_u not in recv_map[part_v]:
            recv_map[part_v][part_u] = []
        recv_map[part_v][part_u].append(v.item())

# 将send_map和recv_map转换为张量列表，方便后续处理
for part in send_map:
    for target_part in send_map[part]:
        send_map[part][target_part] = torch.tensor(send_map[part][target_part])

for part in recv_map:
    for source_part in recv_map[part]:
        recv_map[part][source_part] = torch.tensor(recv_map[part][source_part])
from all_to_all import all_to_all
# 初始化分布式环境
def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29515'
    dist.init_process_group(backend, rank=rank, world_size=size, init_method='env://')
    fn(rank, size)
# from buffer import LocalFeatureStore
# feature_store = LocalFeatureStore()
# 第4步：定义跨多分区消息传递的 GCN 层

# 第5步：运行带有跨分区消息传递的GCN
def run(rank, size):
    print(f"Rank {rank}: 进入run函数")
    # processing data
    # for data(graph structure & graph feature) in dataset:
    
    gcn_layer = GCNLayer(in_feats=3, out_feats=3, num_parts=num_parts)
    output = gcn_layer.forward(graph, feat, send_map, recv_map, rank, size)
    

    print(f"Rank {rank} 的输出特征：")
    print(output)

if __name__ == "__main__":
    size = 4  # 使用4个进程 (每个GPU一个子图)
    torch.multiprocessing.set_start_method('spawn')
    processes = []

    for rank in range(size):
        p = torch.multiprocessing.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

