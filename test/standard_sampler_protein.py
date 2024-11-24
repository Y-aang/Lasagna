import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.optim as optim
from dgl.data import TUDataset
from dgl.nn import GraphConv
import dgl.function as fn

class SimpleGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(SimpleGCN, self).__init__()
        self.gcn1 = nn.Linear(in_feats, hidden_feats)
        nn.init.constant_(self.gcn1.weight, 1)
        nn.init.constant_(self.gcn1.bias, 1)
        
        self.gcn2 = nn.Linear(hidden_feats, out_feats)
        nn.init.constant_(self.gcn2.weight, 1)
        nn.init.constant_(self.gcn2.bias, 1)

    def forward(self, graph, features):
        with graph.local_scope():
            in_degrees = graph.in_degrees().float().clamp(min=1)  # 避免除以0
            norm = torch.pow(in_degrees, -0.5)  # 计算 1/sqrt(in_degree)
            norm = norm.to(features.device).unsqueeze(1)  # 保证维度对齐
            
            graph.ndata['h'] = features * norm  # 节点特征归一化
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = graph.ndata['h']
            h = self.gcn1(h)
            h = torch.relu(h)  # 激活函数
            
            graph.ndata['h'] = h * norm  # 节点特征归一化
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = graph.ndata['h']
            h = self.gcn2(h)
            h = torch.relu(h)
            
            return h
    
dataset = TUDataset(name='PROTEINS', raw_dir=None)
dataset = Subset(dataset, range(2))

for graph, tag in dataset:
    graph.ndata['tag'] = tag.repeat(graph.num_nodes(), 1).float()

graphs = [
    dgl.add_self_loop(graph[0]) for graph in dataset
]

def collate_fn(batch):
    # batch 中的每个元素是一个 (graph, label) 对
    # graphs, labels = zip(*batch)  # 解包
    return batch

train_loader = DataLoader(
    graphs, batch_size=1, shuffle=True, collate_fn=collate_fn
)

model = SimpleGCN(in_feats=1, hidden_feats=3, out_feats=1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.L1Loss(reduction='sum')

# 遍历 DataLoader
optimizer.zero_grad()  # 初始化梯度
outputs = []
losses = []

print("检查模型参数：")
for name, param in model.named_parameters():
    print(f"{name}: {param}")

for batch in train_loader:
    graph = batch[0]
    feat = graph.ndata['node_attr'].to(torch.float32)
    tag = graph.ndata['node_labels'].to(torch.float32)
    output = model(graph, feat)
    print(f"Output for graph(图节点数量：{len(feat)}): {output}")
    loss = loss_fn(output, tag)
    losses.append(loss)
    outputs.append(output)
    loss.backward(retain_graph=True)  # 累积梯度
    print("检查模型参数：")
    for name, param in model.named_parameters():
        print(f"{name}: {param}")

print("losses: ", losses)

# 更新参数
optimizer.step()

# 打印更新后的模型参数
print("检查模型参数：")
for name, param in model.named_parameters():
    print(f"{name}: {param}")
