import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl.data import TUDataset
from dgl.nn import GraphConv

class GCNRegression(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCNRegression, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, activation=F.relu)
        self.conv2 = GraphConv(hidden_feats, hidden_feats, activation=F.relu)
        self.conv3 = GraphConv(hidden_feats, hidden_feats, activation=F.relu)
        self.conv4 = GraphConv(hidden_feats, hidden_feats, activation=F.relu)
        self.conv5 = GraphConv(hidden_feats, hidden_feats, activation=F.relu)
        self.conv6 = GraphConv(hidden_feats, out_feats)
        self.linear = nn.Linear(out_feats, 1)
        
    def forward(self, g, features):
        h = self.conv1(g, features)
        h = self.conv2(g, h)
        h = self.conv3(g, h)
        h = self.conv4(g, h)
        h = self.conv5(g, h)
        h = self.conv6(g, h)
        g.ndata['h'] = h
        output = self.linear(h)
        return output
    
dataset = TUDataset(name='PROTEINS', raw_dir=None)

for graph, tag in dataset:
    graph.ndata['tag'] = tag.repeat(graph.num_nodes(), 1).float()

graphs = [
    dgl.add_self_loop(graph[0]) for graph in dataset
]

train_ratio = 0.8
train_size = int(len(graphs) * train_ratio)
val_size = len(graphs) - train_size
train_graphs = graphs[:train_size]
val_graphs = graphs[train_size:]


def collate_fn(batch):
    # batch 中的每个元素是一个 (graph, label) 对
    # graphs, labels = zip(*batch)  # 解包
    return batch

train_loader = DataLoader(
    train_graphs, batch_size=1, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_graphs, batch_size=1, shuffle=False, collate_fn=collate_fn
)

model = GCNRegression(in_feats=1, hidden_feats=16, out_feats=16)  # 假设节点有 3 维特征
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

# 模型训练函数
def train(model, train_loader, val_loader, epochs=50):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for graphs in train_loader:
            # 随机初始化节点特征
            graph = graphs[0]
            labels = graph.ndata['tag'].float()
            features = graph.ndata['node_attr'].float()
            pred = model(graph, features)
            loss = loss_fn(pred, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for graphs in val_loader:
                graph = graphs[0]
                labels = graph.ndata['tag'].float()
                features = graph.ndata['node_attr'].float()
                pred = model(graph, features)
                loss = loss_fn(pred, labels)
                val_loss += loss.item()
            val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 训练模型
train(model, train_loader, val_loader, epochs=50)
