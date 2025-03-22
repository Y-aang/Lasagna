import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
from torch_geometric.loader import DataLoader
from utils.pyg_dataset import pyg_dataset
from layer import DGLGNN_node

def convert_pyg_to_dgl(data, device):
    """
    将 PyG 的 Data 对象转换为符合下面格式的 batched_data 字典：
      {
          'g': DGLGraph,
          'x': 实例节点特征张量,      # 形状为 [num_instances, feat_dim]
          'x_net': 网络节点特征张量,  # 形状为 [num_net_nodes, feat_dim_net]
          'num_instances': 实例节点数
      }
    data 中需包含：
      - data.edge_index 或 data.edge_index_node_net：边索引
      - data.x：实例节点特征
      - data.x_net：网络节点特征
    """
    # 优先使用 edge_index 字段，否则使用 edge_index_node_net
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        edge_index = data.edge_index
    elif hasattr(data, 'edge_index_node_net') and data.edge_index_node_net is not None:
        edge_index = data.edge_index_node_net
    else:
        raise ValueError("未找到有效的边索引字段 (edge_index 或 edge_index_node_net)")
    
    # 计算总节点数：若存在 num_instances，则总节点数 = 实例节点数 + 网络节点数
    if hasattr(data, 'num_instances'):
        num_instances = data.num_instances
        num_nodes = num_instances + data.x_net.size(0)
    else:
        num_nodes = data.x.size(0) + data.x_net.size(0)
    
    # 构造 DGLGraph
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    
    batched_data = {
        'g': g.to(device),
        'x': data.x.float().to(device),      # 转换为 Float 并移到 device
        'x_net': data.x_net.float().to(device),
        'num_instances': data.x.size(0)
    }
    return batched_data

# 数据路径及图索引（请根据实际情况修改）
data_dir = '/data/yangshen/template/chips/de_hnn/data/2023-03-06_data'
graph_index = 3

# 构造数据集（pyg_dataset 的代码保持不变）
dataset = pyg_dataset(
    data_dir=data_dir, 
    graph_index=graph_index, 
    target='hpwl', 
    load_pe=True, 
    num_eigen=10, 
    load_global_info=False, 
    load_pd=True, 
    vn=False, 
    net=True, 
    split=1, 
    pl=0
)
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 选择设备，若有 GPU 则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 真实数据中，x 的形状为 [797938, 35]，x_net 的形状为 [821523, 1]，
# 所以在构造模型时，将 instance_in_dim 设置为 35，net_in_dim 设置为 1。
model = DGLGNN_node(
    num_layer=2,       # GNN 层数
    emb_dim=16,        # 嵌入维度
    instance_in_dim=35,  # 实例节点输入维度
    net_in_dim=1,        # 网络节点输入维度
    JK="last",         # JK 连接方式
    residual=True,
    gnn_type='gcn',
    norm_type="layer"
).to(device)

# 定义 classifier，并移到设备上
classifier = nn.Linear(16, 1).to(device)

optimizer = optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr=0.01
)

num_epoch = 5

for epoch in range(num_epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, data in enumerate(dataloader):
        # 将 PyG Data 对象转换为 DGL 格式，同时将数据移到 device
        batched_data = convert_pyg_to_dgl(data, device)
        optimizer.zero_grad()
        # 模型前向传播，返回网络节点表示，形状为 [num_net_nodes, 16]
        net_repr = model(batched_data)
        # 真实标签存放在 data.y，假设形状为 [num_net_nodes, 1]，需移到 device
        target = data.y.float().to(device)
        pred = classifier(net_repr)
        loss = F.mse_loss(pred.view(-1), target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Epoch {epoch}, Batch {batch_idx}, Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / (batch_idx + 1)
    print(f"Epoch {epoch} Average Loss = {avg_loss:.4f}")

print("Training finished.")
