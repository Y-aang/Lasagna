import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.data import TUDataset
from dgl.partition import metis_partition
import torch.distributed as dist
import os, sys
import torch.nn.init as init
import copy
# from all_to_all import all_to_all

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from layer import GCNLayer

def add_external_nodes_and_edges(part_id, subgraph, recv_map, global_to_local_maps, original_graph, feature_dim=3):
    # 获取子图的全局节点索引
    global_node_ids = subgraph.ndata['_ID'].tolist()
    
    # 获取 recv_map 中的外来节点列表
    external_nodes = []
    for nodes in recv_map.get(part_id, {}).values():
        external_nodes.extend(nodes)
    external_nodes = list(set(external_nodes))  # 去重

    # 添加外来节点到子图中
    num_external_nodes = len(external_nodes)
    subgraph.add_nodes(num_external_nodes)
    
    # 更新 _ID 字段，记录全局索引
    new_global_node_ids = global_node_ids + external_nodes
    subgraph.ndata['_ID'] = torch.tensor(new_global_node_ids)

    # 添加外来节点的边
    external_edges_src = []
    external_edges_dst = []

    for external_node in external_nodes:
        # 获取外来节点的所有入边
        src_nodes_in, dst_nodes_in = original_graph.in_edges(external_node)
        # 获取外来节点的所有出边
        src_nodes_out, dst_nodes_out = original_graph.out_edges(external_node)
        
        # 合并入边和出边
        for src, dst in zip(src_nodes_in, dst_nodes_in):
            if src.item() in global_node_ids or dst.item() in global_node_ids:
                external_edges_src.append(src.item())
                external_edges_dst.append(dst.item())
        
        for src, dst in zip(src_nodes_out, dst_nodes_out):
            if src.item() in global_node_ids or dst.item() in global_node_ids:
                external_edges_src.append(src.item())
                external_edges_dst.append(dst.item())
    # 转换为接受序号
    local_recv_map_rank = global_to_local_maps[part_id]
    external_edges_src = [local_recv_map_rank.get(node) for node in external_edges_src]
    external_edges_dst = [local_recv_map_rank.get(node) for node in external_edges_dst]

    
    # 将边添加到子图中
    subgraph.add_edges(external_edges_src, external_edges_dst)

def update_global_to_local_maps(global_to_local_maps, recv_map):
    for rank, sub_map in recv_map.items():
        # Gather all unique nodes in the current rank's sub_map
        nodes = set()
        for target_rank, node_list in sub_map.items():
            nodes.update(node_list)

        # Sort the nodes to maintain a consistent order
        nodes = sorted(nodes)

        # Find the next available local index for the current rank
        current_max_index = max(global_to_local_maps[rank].values(), default=-1)
        next_index = current_max_index + 1

        # Add new nodes to the global_to_local_maps for the current rank
        for node in nodes:
            node = node.item()
            if node not in global_to_local_maps[rank]:
                global_to_local_maps[rank][node] = next_index
                next_index += 1
    return global_to_local_maps

def convert_map_to_local(recv_map, global_to_local_maps):
    # Create a deep copy of recv_map to avoid modifying the original one
    local_recv_map = copy.deepcopy(recv_map)
    
    for rank, sub_map in local_recv_map.items():
        # Iterate over the target ranks and node lists in the current rank's sub-map
        for target_rank, nodes in sub_map.items():
            # Update the nodes using global_to_local_maps
            local_recv_map[rank][target_rank] = [global_to_local_maps[rank].get(node.item()) for node in nodes]

    return local_recv_map






# 第1步：加载TUDataset中的PROTEINS数据集
# dataset = TUDataset(name='PROTEINS')
# graph, label = dataset[0]
# print(f"图信息：{graph}")

num_nodes = 12
# edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]  # 形成圆环
edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)] + [((i + 1) % num_nodes, i) for i in range(num_nodes)]  # 形成圆环
u, v = zip(*edges)
graph = dgl.graph((u, v))
feat = torch.tensor([
    [1, 1, 1],
    [1, 2, 1],
    [1, 3, 1],
    [1, 4, 1],
    [1, 5, 1],
    [1, 6, 1],
    [1, 7, 1],
    [1, 8, 1],
    [1, 9, 1],
    [1, 10, 1],
    [1, 11, 1],
    [1, 12, 1]
], dtype=torch.float)
tag = torch.tensor([
    [1], [2], [3], [4], [5], [6],
    [7], [8], [9], [10], [11], [12]
], dtype=torch.float)
tag = torch.tensor([
    [1, 2, 1],
    [1, 3, 1],
    [1, 4, 1],
    [1, 5, 1],
    [1, 6, 1],
    [1, 7, 1],
    [1, 8, 1],
    [1, 9, 1],
    [1, 10, 1],
    [1, 11, 1],
    [1, 12, 1],
    [1, 13, 1]
], dtype=torch.float)

graph.ndata['h'] = feat
graph.ndata['tag'] = tag
print(graph.ndata)

# 第2步：使用 Metis 将图分成 n 部分
num_parts = 4  # 假设我们将图分成4部分
parts = metis_partition(graph, k=num_parts)
global_to_local_maps = {}

# 提取每个节点的分区编号（通过part_id）
node_part = torch.empty(num_nodes, dtype=torch.int64)

# 处理分布式特征和节点序号 global_to_local_maps
for part_id, subgraph in parts.items():
    global_node_ids = subgraph.ndata['_ID']
    subgraph.ndata['h'] = feat[global_node_ids]
    subgraph.ndata['tag'] = tag[global_node_ids]
    global_to_local = {global_id.item(): local_id for local_id, global_id in enumerate(global_node_ids)}
    global_to_local_maps[part_id] = global_to_local

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
        if u.item() not in send_map[part_u][part_v]:
            send_map[part_u][part_v].append(u.item())
        # v 需要从 u 的子图接收特征
        if part_u not in recv_map[part_v]:
            recv_map[part_v][part_u] = []
        if u.item() not in recv_map[part_v][part_u]:
            recv_map[part_v][part_u].append(u.item())

# 将send_map和recv_map转换为张量列表，方便后续处理
for part in send_map:
    for target_part in send_map[part]:
        send_map[part][target_part] = torch.tensor(send_map[part][target_part])

for part in recv_map:
    for source_part in recv_map[part]:
        recv_map[part][source_part] = torch.tensor(recv_map[part][source_part])

# 更新global_to_local_maps
global_to_local_maps = update_global_to_local_maps(global_to_local_maps, recv_map)

# 为每个子图添加外来节点和边
for part_id, subgraph in parts.items():
    add_external_nodes_and_edges(part_id, subgraph, recv_map, global_to_local_maps, graph)

# 获取local的recv和send map用于作为dataset的处理后输出
local_send_map = convert_map_to_local(send_map, global_to_local_maps)
local_recv_map = convert_map_to_local(recv_map, global_to_local_maps)



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
    
    # TODO: gain data from path
    
    gcn_layer = GCNLayer(in_feats=3, out_feats=3, num_parts=num_parts)
    output = gcn_layer.forward(parts[rank], parts[rank].ndata['h'], local_send_map, local_recv_map, rank, size)

    print(f"Rank {rank} 的输出特征：")
    print(output)
    print('节点 target', parts[rank].ndata['tag'])

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

