


import dgl
import torch
from dgl.partition import metis_partition


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
graph.ndata['h'] = feat
graph.ndata['tag'] = tag
print('1: ', graph.ndata)


num_parts = 4  # 假设我们将图分成4部分
parts = metis_partition(graph, k=num_parts)

print('2: ', parts[0].ndata)

