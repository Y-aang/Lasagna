import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from mlpgate import MLPGateDGL
from dgl.data.utils import load_graphs

torch.manual_seed(42)  # 固定种子

# Step 1: Load the dataset from the .pt file
dataset_path = '/data/yangshen/workplace/Lasagna/test/Deepgate/data/graphs_data.pt'  # 保存的 .pt 文件路径
data_list = torch.load(dataset_path)  # 从 .pt 文件中加载所有图

batch_size = 4
dataloader = dgl.dataloading.GraphDataLoader(data_list, batch_size=batch_size, shuffle=True)

# Adjust input feature dimensions
args = {
    'dim_node_feature': 32,   # 输入特征维度
    'dim_hidden': 4,         # 模型中隐藏层维度
    'dim_mlp': 8,           # MLP 层的维度
    'num_rounds': 1,          # 循环次数
    'device': 'cuda',         # 设备类型
}

# Initialize model, optimizer, and loss function
model = MLPGateDGL(args).to(args['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Step 2: Train the model with all graphs
for epoch in range(1):  # 假设训练 10 个 epoch
    total_loss = 0.0
    total_loss = 0.0
    for batch_id, batched_graph in enumerate(dataloader):
        # Move batch to GPU
        batched_graph = batched_graph.to(args['device'])

        # 提取特征和标签
        labels = batched_graph.ndata['prob'].squeeze()  # 假设标签存储在 'prob' 中
        rc_pair_index = batched_graph.edata.get('rc_pair_index', None)  # 获取 rc_pair_index，可能不存在
        if rc_pair_index is not None:
            rc_pair_index = rc_pair_index.to(args['device'])

        # Forward pass
        hs, hf, prob, is_rc = model(batched_graph, rc_pair_index=rc_pair_index)

        # Compute loss
        loss = criterion(prob.squeeze(), labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Batch {batch_id + 1}/{len(dataloader)}, Loss: {loss.item()}")

    print(f"Epoch {epoch + 1} Total Loss: {total_loss:.4f}")