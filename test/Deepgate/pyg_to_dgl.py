import dgl
import torch

def pyg_to_dgl(G):
    # 创建 DGL 图
    edge_index = G.edge_index
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=G.num_nodes)
    
    # 添加节点特性
    dgl_graph.ndata['x'] = G.x
    dgl_graph.ndata['gate'] = G.gate
    dgl_graph.ndata['forward_level'] = G.forward_level
    dgl_graph.ndata['backward_level'] = G.backward_level

    # 添加边特性
    if G.edge_attr is not None:
        dgl_graph.edata['edge_attr'] = G.edge_attr

    # 添加 RC 信息
    dgl_graph.ndata['rc_pair_index'] = G.rc_pair_index

    return dgl_graph
