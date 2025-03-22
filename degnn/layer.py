import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class DGLGCNConv(nn.Module):
    def __init__(self, emb_dim, edge_dim=None):
        """
        emb_dim: 节点特征维度（输入和输出均为 emb_dim）
        edge_dim: 边特征维度（此处未用到，可根据需要扩展）
        """
        super(DGLGCNConv, self).__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)

    def forward(self, g, x):
        """
        g: DGLGraph
        x: 节点特征张量，形状为 (N, emb_dim)
        """
        # 1. 对节点特征作线性变换
        x = self.linear(x)
        # 2. 计算节点度（加上自环效果）
        deg = g.in_degrees().float() + 1.0  # shape: (N,)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        g.ndata['deg_inv_sqrt'] = deg_inv_sqrt
        # 3. 保存线性变换后的特征到节点数据中
        g.ndata['h'] = x
        # 4. 计算每条边的归一化系数： norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        g.apply_edges(lambda edges: {'norm': edges.src['deg_inv_sqrt'] * edges.dst['deg_inv_sqrt']})
        # 5. 使用 update_all 完成消息传递
        def message_func(edges):
            return {'m': edges.data['norm'].unsqueeze(-1) * F.relu(edges.src['h'])}
        def reduce_func(nodes):
            return {'agg': torch.sum(nodes.mailbox['m'], dim=1)}
        g.update_all(message_func, reduce_func)
        agg = g.ndata.pop('agg')
        # 6. 自环更新
        self_loop = F.relu(x + self.root_emb.weight) * (1.0 / deg.view(-1, 1))
        # 7. 返回消息传递与自环更新之和
        return agg + self_loop

class DGLGNN_node(nn.Module):
    """
    基于 DGL 的节点级 GNN 模型（这里只实现了 gcn 类型）。
    输入 batched_data 为一个字典，包含：
      - 'g': DGLGraph（图结构）
      - 'x': 实例节点特征张量
      - 'x_net': 网络节点特征张量
      - 'num_instances': 实例节点数
    构造函数增加了：
      - instance_in_dim：实例节点原始特征维度（默认 11）
      - net_in_dim：网络节点原始特征维度（默认 3）
    """
    def __init__(self, num_layer, emb_dim, instance_in_dim=11, net_in_dim=3, JK="concat", residual=True, gnn_type='gcn', norm_type="layer"):
        super(DGLGNN_node, self).__init__()
        self.num_layer = num_layer
        self.JK = JK
        self.residual = residual
        self.emb_dim = emb_dim
        self.gnn_type = gnn_type
        
        if self.gnn_type == 'gcn':
            self.convs = nn.ModuleList([DGLGCNConv(emb_dim) for _ in range(num_layer)])
            self.re_convs = nn.ModuleList([DGLGCNConv(emb_dim) for _ in range(num_layer)])
        else:
            raise NotImplementedError("目前只示例了 gcn 类型的 DGL 版本。")
        
        self.norms = nn.ModuleList()
        for _ in range(num_layer):
            if norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(emb_dim))
            elif norm_type == "layer":
                self.norms.append(nn.LayerNorm(emb_dim))
            else:
                raise NotImplementedError("未实现该归一化类型")
        
        # 节点编码器：将实例节点特征从 instance_in_dim 映射到 emb_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(instance_in_dim, emb_dim * 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LeakyReLU(negative_slope=0.1)
        )
        # 网络节点编码器：将网络节点特征从 net_in_dim 映射到 emb_dim
        self.node_encoder_net = nn.Sequential(
            nn.Linear(net_in_dim, emb_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(negative_slope=0.1)
        )
    
    def forward(self, batched_data):
        """
        输入 batched_data 为一个字典，包含：
          - 'g': DGLGraph
          - 'x': 实例节点特征张量
          - 'x_net': 网络节点特征张量
          - 'num_instances': 实例节点数
        """
        g = batched_data['g']
        # 对实例节点和网络节点分别编码
        x_inst = self.node_encoder(batched_data['x'])
        x_net = self.node_encoder_net(batched_data['x_net'])
        num_instances = batched_data['num_instances']
        
        # 拼接成一个长向量：前 num_instances 为实例节点，后面为网络节点
        x = torch.cat([x_inst, x_net], dim=0)
        h_list = [x]
        
        for layer in range(self.num_layer):
            # 构造反向图，用于双向消息传递
            g_rev = g.reverse(copy_ndata=True, copy_edata=True)
            h_forward = self.convs[layer](g, h_list[layer])
            h_reverse = self.re_convs[layer](g_rev, h_list[layer])
            
            h = h_forward + h_reverse
            h = self.norms[layer](h)
            h = F.leaky_relu(h, negative_slope=0.1)
            if self.residual:
                h = h + h_list[layer]
            h_list.append(h)
        
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = sum(h_list)
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        else:
            raise NotImplementedError("未实现该 JK 连接方式")
        
        # 返回网络节点部分的表示（假设前 num_instances 为实例节点）
        return node_representation[num_instances:]
