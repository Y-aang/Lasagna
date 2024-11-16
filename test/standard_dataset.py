import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.data import TUDataset
from dgl.nn import GraphConv
import dgl
from sklearn.metrics import accuracy_score

# 1. 加载 PROTEINS 数据集
dataset = TUDataset(name='PROTEINS')

# 检查第一个图的信息
graph, label = dataset[0]
print("Node attributes:", graph.ndata['node_attr'])
print("Graph label:", label)

# 预处理：将 'node_attr' 作为节点特征
for graph, label in dataset:
    graph = dgl.add_self_loop(graph)
    graph.ndata['feat'] = graph.ndata.pop('node_attr').float()

# 2. 定义自定义批量采样函数
def collate_fn(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels

# 数据集划分为训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# 3. 定义 GCN 模型
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, activation=F.relu)
        self.conv2 = GraphConv(hidden_feats, out_feats)
        self.classifier = nn.Linear(out_feats, dataset.num_classes)
    
    def forward(self, g, features):
        h = self.conv1(g, features)
        h = self.conv2(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')  # 图池化
            return self.classifier(hg)

# 4. 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_feats=dataset[0][0].ndata['feat'].shape[1], hidden_feats=32, out_feats=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    total_loss = 0
    for batched_graph, labels in train_loader:
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata['feat'].to(device)
        labels = labels.to(device)
        
        logits = model(batched_graph, features)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch + 1}, Loss: {total_loss:.4f}')

# 5. 评估模型
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batched_graph, labels in test_loader:
        batched_graph = batched_graph.to(device)
        features = batched_graph.ndata['feat'].to(device)
        labels = labels.to(device)
        
        logits = model(batched_graph, features)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy:.4f}')
