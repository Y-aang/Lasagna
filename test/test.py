import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.data import TUDataset
from dgl.partition import metis_partition
import torch.distributed as dist
import os

# 第1步：加载TUDataset中的PROTEINS数据集
dataset = TUDataset(name='PROTEINS')

# 取出第一个图和对应的标签
graph, label = dataset[0]
print(f"图信息：{graph}")

# 初始化节点特征
num_nodes = graph.num_nodes()
feat = torch.randn(num_nodes, 3)  # 假设每个节点有3维特征
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
        send_map[part_u][part_v].append(u.item())

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
def communicate_grad(grad: torch.Tensor):
    # start to send the grad and retrieve
    self_rank = dist.get_rank()
    world_size = dist.get_world_size()
    recv_list = [torch.tensor([0]) if i == self_rank else torch.empty_like(feat[send_map[self_rank][i]]) for i in range(world_size)]
    send_list = [torch.tensor([0]) if i == self_rank else grad[recv_map[self_rank][i]] for i in range(world_size)]
    
    all_to_all(recv_list, send_list)
    
    for i in range(world_size):
        if i == self_rank:
            continue
        grad[send_map[self_rank][i]] += recv_list[i]
    return grad
class GCNLayerWithPartition(nn.Module):
    def __init__(self, in_feats, out_feats, num_parts):
        super(GCNLayerWithPartition, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.num_parts = num_parts

    def forward(self, graph, feat, send_map, recv_map, rank, size):
        # 1. 跨GPU消息传递：发送节点特征到接收节点
        dist.barrier()  # 等待所有进程同步
        print(f"Rank {rank}: 进入消息传递阶段")
        ops = []
        feat.register_hook(communicate_grad)
        send_list = [torch.tensor([0.0])] * size
        
        for part_v in send_map[rank]:
            # 发送特征到其他GPU
            send_feat = feat[send_map[rank][part_v]].clone()
            print(f"Rank {rank}: 发送特征到 {part_v}, send_feat.shape={send_feat.shape}")
            # dist.isend(tensor=send_feat, dst=part_v)
            send_list[part_v] = send_feat
            # print("send:", dist.isend(tensor=send_feat, dst=part_v))
        output = [torch.empty(1)] * size
        for part_v in recv_map[rank]:
            # 接收特征来自其他GPU
            recv_feat = torch.empty_like(feat[recv_map[rank][part_v]])
            output[part_v] = recv_feat
        # dist.all_to_all(output, send_list)
        print('Go Send')
        all_to_all(output, send_list)
        # dist.barrier()
        print(f"Rank {rank}: 进入消息接受阶段")
        for part_v in recv_map[rank]:
            # 接收特征来自其他GPU
            recv_feat = output[part_v]
            # dist.irecv(tensor=recv_feat, src=part_v).wait()
            # 更新接收节点特征
            feat[recv_map[rank][part_v]] = recv_feat
            print(f"Rank {rank}: 接收特征来自 {part_v}, recv_feat.shape={recv_feat.shape}")
        print(f"Rank {rank}: Finish")
        dist.barrier()  # 再次同步，确保所有进程完成通信

        print(f'Rank {rank}: process')
        
        # 2. 消息传递和特征聚合 (每个子图内进行)
        graph.ndata['h'] = feat
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))

        # 3. 获取聚合后的特征
        h = graph.ndata.pop('h')

        # 4. 线性变换
        h = self.linear(h)
        return h

# 第5步：运行带有跨分区消息传递的GCN
def run(rank, size):
    print(f"Rank {rank}: 进入run函数")
    gcn_layer = GCNLayerWithPartition(in_feats=3, out_feats=2, num_parts=num_parts)
    output = gcn_layer(graph, feat, send_map, recv_map, rank, size)

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

