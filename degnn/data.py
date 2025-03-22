from torch_geometric.loader import DataLoader
from utils.pyg_dataset import pyg_dataset
import dgl

def convert_pyg_to_dgl(data):
    """
    将 PyG 的 Data 对象转换为符合下面格式的 batched_data 字典：
      {
          'g': DGLGraph,
          'x': 实例节点特征张量,      # 形状为 [num_instances, feat_dim]
          'x_net': 网络节点特征张量,  # 形状为 [num_net_nodes, feat_dim_net]
          'num_instances': 实例节点数
      }
    假设 data 中已经包含：
      - data.edge_index 或 data.edge_index_node_net：边索引，形状 [2, num_edges]
      - data.x：实例节点的特征
      - data.x_net：网络节点的特征
    """
    # 优先使用 edge_index 字段，否则使用 edge_index_node_net
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        edge_index = data.edge_index
    elif hasattr(data, 'edge_index_node_net') and data.edge_index_node_net is not None:
        edge_index = data.edge_index_node_net
    else:
        raise ValueError("未找到有效的边索引字段 (edge_index 或 edge_index_node_net)")

    # 计算图中总节点数：若有 num_instances 则取其加上网络节点数，否则用 x 与 x_net 数量之和
    if hasattr(data, 'num_instances'):
        num_instances = data.num_instances
        num_nodes = num_instances + data.x_net.size(0)
    else:
        num_nodes = data.x.size(0) + data.x_net.size(0)
    
    # 构造 DGLGraph
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    
    # 构造转换后的字典
    batched_data = {
        'g': g,
        'x': data.x,                # 实例节点特征
        'x_net': data.x_net,        # 网络节点特征
        'num_instances': data.x.size(0)  # 假设 data.x 中的所有节点都是实例节点
    }
    return batched_data

# 以下部分保持不变，从 pyg_dataset 构造数据并测试转换

data_dir = '/data/yangshen/template/chips/de_hnn/data/2023-03-06_data'
dataset = pyg_dataset(data_dir=data_dir, graph_index=3, target='hpwl', load_pe=True, num_eigen=10, load_global_info=False, load_pd=True, vn=False, net=True, split=1, pl=0)
batch_size = 1
dataloader = DataLoader(dataset, batch_size, shuffle=False)

# 示例：在 dataloader 循环中进行转换
for batch_idx, data in enumerate(dataloader):
    # data 是 PyG 的 Data 对象
    batched_data = convert_pyg_to_dgl(data)
    
    # 打印转换后的信息，方便检查
    print("Batch", batch_idx)
    print("DGLGraph:", batched_data['g'])
    print("Instance node features shape:", batched_data['x'].shape)
    print("Network node features shape:", batched_data['x_net'].shape)
    print("Num instances:", batched_data['num_instances'])
    
    # 之后可以直接将 batched_data 送入 DGL 的模型
    # output = model(batched_data)
