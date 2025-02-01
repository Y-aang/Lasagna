import os
import numpy as np
import numpy
import torch
import dgl
import math


def read_npz_file(filename, dir):
    path = os.path.join(dir, filename)
    data = np.load(path, allow_pickle=True)
    return data

def construct_node_feature_dgl(x, num_gate_types, no_node_cop=False, node_reconv=False):
    """
    构建节点特征，转换为 DGL 兼容格式。
    Args:
        x: 原始节点特征矩阵（numpy array），形状 [num_nodes, num_features]。
        num_gate_types: 门类型的数量，用于 one-hot 编码。
        no_node_cop: 是否忽略节点特定的特征。
        node_reconv: 是否添加 `node_reconv` 信息。
    Returns:
        x_dgl: 转换后的节点特征张量，适用于 DGL。
    """
    # 提取 gate 类型并进行 one-hot 编码
    gate_list = x[:, 1]  # 假设 gate 类型存储在 x 的第 1 列
    gate_list = np.float32(gate_list)
    x_dgl = torch.nn.functional.one_hot(torch.tensor(gate_list, dtype=torch.long), num_classes=num_gate_types).float()
    # 如果启用 node_reconv，添加 reconvergence 特征
    # if node_reconv:
    #     reconv = torch.tensor(x[:, 7], dtype=torch.float).unsqueeze(1)  # 假设第 7 列是 reconv 信息
    #     x_dgl = torch.cat([x_dgl, reconv], dim=1)
    return x_dgl

def add_edge_attr(num_edge, ehs, ll_diff=1):
    positional_embeddings = torch.zeros(num_edge, ehs)
    for position in range(num_edge):
        for i in range(0, ehs, 2):
            positional_embeddings[position, i] = (
                math.sin(ll_diff / (10000 ** ((2 * i) / ehs)))
            )
            positional_embeddings[position, i + 1] = (
                math.cos(ll_diff / (10000 ** ((2 * (i + 1)) / ehs)))
            )

    return positional_embeddings

def top_sort(edge_index, graph_size):
    node_ids = numpy.arange(graph_size, dtype=int)

    node_order = numpy.zeros(graph_size, dtype=int)
    unevaluated_nodes = numpy.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]

        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    return torch.from_numpy(node_order).long()

def return_order_info(edge_index, num_nodes):
    ns = torch.LongTensor([i for i in range(num_nodes)])
    forward_level = top_sort(edge_index, num_nodes)
    ei2 = torch.LongTensor([list(edge_index[1]), list(edge_index[0])])
    backward_level = top_sort(ei2, num_nodes)
    forward_index = ns
    backward_index = torch.LongTensor([i for i in range(num_nodes)])
    
    return forward_level, forward_index, backward_level, backward_index

def parse_pyg_mlpgate(x, edge_index, tt_dis, min_tt_dis, tt_pair_index, y, rc_pair_index, is_rc, 
                      use_edge_attr=False, reconv_skip_connection=False, no_node_cop=False, 
                      node_reconv=False, un_directed=False, num_gate_types=9, dim_edge_feature=32):
    """
    将输入特征和边索引转换为 DGL 图。
    Args:
        x: 节点特征 (numpy array)。
        edge_index: 边索引，形状为 [2, num_edges]。
        tt_dis, min_tt_dis, tt_pair_index, y, rc_pair_index, is_rc: 其他图相关特征。
        un_directed: 是否将图转换为无向图。
    Returns:
        graph: DGL 图对象，包含节点和边的特征。
    """
    # 构建节点特征
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = add_edge_attr(len(edge_index), dim_edge_feature, 1)
    edge_index = edge_index.t().contiguous()
    x_dgl = construct_node_feature_dgl(x, num_gate_types, no_node_cop, node_reconv)
    edge_src, edge_dst = edge_index[0], edge_index[1]
    graph = dgl.graph((edge_src, edge_dst), num_nodes=x_dgl.size(0))
    graph.ndata['feat'] = x_dgl
    
    if reconv_skip_connection:      # False
        edge_index, edge_attr = add_skip_connection(x, edge_index, edge_attr, dim_edge_feature)

    # graph.ndata['is_rc'] = torch.tensor(is_rc, dtype=torch.float32).unsqueeze(1)
    graph.ndata['gate'] = torch.tensor(x[:, 1:2], dtype=torch.float).squeeze(dim=1)
    graph.ndata['prob'] = torch.tensor(y).reshape((len(x), 1))
    # graph.ndata['tt_dis'] = torch.tensor(tt_dis, dtype=torch.float32)
    # graph.ndata['min_tt_dis'] = torch.tensor(min_tt_dis, dtype=torch.float32)

    # 添加额外的边特征
    # graph.edata['tt_pair_index'] = torch.tensor(tt_pair_index, dtype=torch.long).t().contiguous()
    # graph.edata['rc_pair_index'] = torch.tensor(rc_pair_index, dtype=torch.long).t().contiguous()
    
    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x_dgl.size(0))
    graph.ndata['forward_level'] = forward_level
    graph.ndata['forward_index'] = forward_index
    graph.ndata['backward_level'] = backward_level
    graph.ndata['backward_index'] = backward_index
    
    if use_edge_attr:
        edge_attr = add_edge_attr(len(edge_index), dim_edge_feature, 1)  # 自定义边特征逻辑
        graph.edata['attr'] = edge_attr

    if un_directed:
        graph = dgl.to_bidirected(graph, copy_ndata=True)

    return graph



processed_paths = '/data/yangshen/workplace/Lasagna/test/Deepgate/data/graphs_data.pt'
data_list = []
tot_pairs = 0
circuit_file = "graphs.npz"
label_file = "labels.npz"
data_dir = "/data/yangshen/template/DeepGate2/data/train"
circuits = read_npz_file(circuit_file, data_dir)['circuits'].item()     # 读的是graph.npz from prepare_dataset.py
labels = read_npz_file(label_file, data_dir)['labels'].item()

for cir_idx, cir_name in enumerate(circuits):
    if cir_idx > 4:
        break
    print(f"Processing item {cir_idx + 1} out of {len(circuits)}")
    print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))
    x = circuits[cir_name]["x"]
    edge_index = circuits[cir_name]["edge_index"]

    tt_dis = labels[cir_name]['tt_dis']
    min_tt_dis = labels[cir_name]['min_tt_dis']
    tt_pair_index = labels[cir_name]['tt_pair_index']
    prob = labels[cir_name]['prob']

    if True:        # self.args.no_rc
        rc_pair_index = [[0, 1]]
        is_rc = [0]
    else:
        rc_pair_index = labels[cir_name]['rc_pair_index']
        is_rc = labels[cir_name]['is_rc']

    if len(tt_pair_index) == 0 or len(rc_pair_index) == 0:
        print('No tt or rc pairs: ', cir_name)
        continue

    tot_pairs += len(tt_dis)

    # check the gate types
    # assert (x[:, 1].max() == (len(self.args.gate_to_index)) - 1), 'The gate types are not consistent.'
    graph = parse_pyg_mlpgate(
        x, edge_index, tt_dis, min_tt_dis, tt_pair_index, prob, rc_pair_index, is_rc,
        num_gate_types=3,        # train 1 用的是 3
        use_edge_attr=False,
        dim_edge_feature=16,
    )
    graph.name = cir_name
    data_list.append(graph)
    # if self.args.small_train and cir_idx > subset:
    #     break

torch.save(data_list, processed_paths)
print('[INFO] Inmemory dataset save: ', processed_paths)
print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))