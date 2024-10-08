import torch
import torch.nn as nn
import dgl
import dgl.function as fn

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

# 创建一个具有 12 个节点的环形图
num_nodes = 12
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
    [1, 12, 1]
], dtype=torch.float32)

# 定义一层简单的 GCN，不进行归一化，只进行特征求和
class SimpleGCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(SimpleGCN, self).__init__()
        # 定义一个线性层
        self.linear = nn.Linear(in_feats, out_feats)
        # 手动初始化权重为全 1
        # nn.init.constant_(self.linear.weight, 1.0)
        nn.init.constant_(self.linear.weight, 1)
        nn.init.constant_(self.linear.bias, 1)

    def forward(self, graph, features):
        with graph.local_scope():
            # 聚合邻居特征：简单求和，不进行归一化
            graph.ndata['h'] = features
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            # 获取聚合后的特征
            h = graph.ndata['h']
            # 应用线性变换
            return self.linear(h)

# 实例化 GCN 模型
gcn = SimpleGCN(in_feats=3, out_feats=3)

# 进行前向传播
with torch.no_grad():
    output = gcn(graph, features)

# 打印输出结果
print("Output features after one GCN layer with weights initialized to 1 (using sum):")
print(output)
