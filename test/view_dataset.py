import dgl
from dgl.data import TUDataset

# 加载MUTAG数据集
dataset = TUDataset(name='MUTAG')

# 获取第一条图数据
graph, label = dataset[0]

# 打印图的节点数
print(f"节点数: {graph.num_nodes()}")

# 在图的每个节点上执行节点分类任务
# 可以根据节点的特征或图的结构来自定义节点分类任务
print(f"节点特征: {graph.ndata}")

