# Import necessary libraries
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from mlpgate import MLPGateDGL

torch.manual_seed(42)  # 固定种子

# Step 1: Create a DGL graph
graph = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]), num_nodes=5)

# Adjust input features to match dim_hidden
args = {
    'dim_node_feature': 32,   # 输入特征维度
    'dim_hidden': 4,         # 模型中隐藏层维度
    'dim_mlp': 8,           # MLP 层的维度
    'num_rounds': 1,          # 循环次数
    'device': 'cuda',         # 设备类型
}
# Adjust the input feature to match dim_hidden
graph.ndata['x'] = torch.zeros(5, args['dim_hidden'])  # Ensure feature dimension matches dim_hidden

# Assign other node data
graph.ndata['gate'] = torch.tensor([0, 1, 1, 0, 0])  # Gate types: 0 = PI, 1 = AND, 2 = NOT
graph.ndata['forward_level'] = torch.tensor([0, 1, 1, 0, 0])  # Forward levels

# Example RC pairs
rc_pair_index = torch.tensor([[0, 1], [2, 3]])

# Binary classification labels
labels = torch.randint(0, 2, (5,)).float()  # Adjust to match the number of nodes

# Step 2: Initialize model, optimizer, and loss function
model = MLPGateDGL(args).to('cuda')
for param in model.parameters():
    param.data.fill_(3.0)
graph = graph.to('cuda')
labels = labels.to('cuda')
rc_pair_index = rc_pair_index.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Step 3: Forward pass
hs, hf, prob, is_rc = model(graph, rc_pair_index=rc_pair_index)

print('hs: ', hs)
print('hf: ', hf)
print('prob: ', prob)
print('is_rc: ', is_rc)

# Step 4: Compute loss
loss = criterion(prob.squeeze(), labels)
# print(f"Loss: {loss.item()}")

# Step 5: Backward pass and optimization
# loss.backward()
# optimizer.step()
