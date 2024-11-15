import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import dgl.function as fn

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

# 创建一个具有 12 个节点的环形图
num_nodes = 13
edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)] + [((i + 1) % num_nodes, i) for i in range(num_nodes)]
u, v = zip(*edges)
graph = dgl.graph((torch.tensor(u), torch.tensor(v)))

# 初始化节点特征
features = torch.tensor([
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
    [1, 12, 1],
    [1, 13, 1],
], dtype=torch.float32)

tag = torch.tensor([
    [19, 19, 19],
    [9, 9, 9],
    [11, 11, 11],
    [13, 13, 13],
    [15, 15, 15],
    [17, 17, 17],
    [19, 19, 19],
    [21, 21, 21],
    [23, 23, 23],
    [25, 25, 25],
    [27, 27, 27],
    [17, 17, 17],
    [17, 17, 17],
], dtype=torch.float)
tag += 1

# 定义一层简单的 GCN，不进行归一化，只进行特征求和
class SimpleGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(SimpleGCN, self).__init__()
        # 第一层 GCN
        self.gcn1 = nn.Linear(in_feats, hidden_feats)
        nn.init.constant_(self.gcn1.weight, 1)
        nn.init.constant_(self.gcn1.bias, 1)
        
        # 第二层 GCN
        self.gcn2 = nn.Linear(hidden_feats, out_feats)
        nn.init.constant_(self.gcn2.weight, 1)
        nn.init.constant_(self.gcn2.bias, 1)

    def forward(self, graph, features):
        with graph.local_scope():
            # 第一层 GCN
            graph.ndata['h'] = features
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = graph.ndata['h']
            h = self.gcn1(h)
            h = torch.relu(h)  # 激活函数
            
            # 第二层 GCN
            graph.ndata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            h = graph.ndata['h']
            h = self.gcn2(h)
            
            return h

# 实例化 GCN 模型
gcn = SimpleGCN(in_feats=3, hidden_feats=3, out_feats=3)

# 进行前向传播
features.requires_grad_(True)
output = gcn(graph, features)
criterion = nn.L1Loss(reduction='sum')
optimizer = optim.SGD(gcn.parameters(), lr=1)
loss = criterion(output, tag)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 打印输出结果
print("Output features after one GCN layer with weights initialized to 1 (using sum):")
print(output)

print("Output weights:")
print(gcn.gcn1.weight, gcn.gcn1.bias)

print("Feat grad:")
print(features.grad)