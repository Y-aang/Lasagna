import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl

from layer import DGLGNN_node

def create_fake_data():
    """
    构造一份虚拟的图数据，模拟原先 PyG Data 的结构，
    并转成 DGLGraph。这里只做演示。
    """
    # 假设有 5 个实例节点和 3 个网络节点
    num_instances = 5
    num_net_nodes = 3
    total_nodes = num_instances + num_net_nodes

    # 定义一些边 (src -> dst)，示例中让前 5 个节点连向后 3 个节点
    src = [0, 1, 2, 3, 4]
    dst = [5, 6, 7, 5, 6]
    g = dgl.graph((src, dst), num_nodes=total_nodes)

    # 实例节点特征：5 个节点，每个 11 维（与 node_encoder 输入维度对应）
    x_inst = torch.randn(num_instances, 11)
    # 网络节点特征：3 个节点，每个 3 维（与 node_encoder_net 输入维度对应）
    x_net = torch.randn(num_net_nodes, 3)

    # 将数据整合到一个字典中
    batched_data = {
        'g': g,                  # DGLGraph
        'x': x_inst,             # 实例节点特征
        'x_net': x_net,          # 网络节点特征
        'num_instances': num_instances
    }
    return batched_data

def main():
    # 构造虚假数据
    batched_data = create_fake_data()
    
    # 定义模型，这里选择 gcn 类型
    model = DGLGNN_node(
        num_layer=2,       # GNN 层数
        emb_dim=16,        # 嵌入维度
        JK="last",         # JK 连接方式，可选 "last"、"sum"、"concat"
        residual=True,
        gnn_type='gcn',
        norm_type="layer"
    )
    
    # 定义一个简单的 classifier，用于将 GNN 输出映射到目标（例如二分类或回归）
    classifier = nn.Linear(16, 1)

    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=0.01
    )

    # 训练循环（示例仅跑 5 个 epoch）
    for epoch in range(5):
        optimizer.zero_grad()
        
        # 模型输出的是网络节点的表示，形状为 [num_net_nodes, 16]
        net_repr = model(batched_data)
        
        # 构造假标签：假设对网络节点做二分类
        num_net_nodes = net_repr.size(0)
        target = torch.randint(0, 2, (num_net_nodes, 1)).float()
        
        # 通过 classifier 得到预测结果
        pred = classifier(net_repr)
        
        # 使用 MSE 作为损失函数（示例）
        loss = F.mse_loss(pred, target)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

if __name__ == "__main__":
    main()
