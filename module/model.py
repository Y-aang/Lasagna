import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn.init as init
from module.layer import GCNLayer
from helper.utils import register_hook_for_model_param

class myGCN(nn.Module):
    def __init__(self, in_feats, out_feats, num_parts):
        super(myGCN, self).__init__()
        self.gcnLayer1 = GCNLayer(in_feats=3, out_feats=2, num_parts=num_parts)
        self.gcnLayer2 = GCNLayer(in_feats=2, out_feats=3, num_parts=num_parts)
    
    # def forward(self, graphStructure, subgraphFeature):
    def forward(self, subgraph, feat, send_map, recv_map, rank, size):
        logits = self.gcnLayer1.forward(subgraph, feat, send_map, recv_map, rank, size)
        logits = self.gcnLayer2.forward(subgraph, logits, send_map, recv_map, rank, size)
        return logits
    
class GCNDataset(nn.Module):
    def __init__(self, in_feats, out_feats, num_parts):
        super(GCNDataset, self).__init__()
        self.gcnLayer1 = GCNLayer(in_feats=1, out_feats=3, num_parts=num_parts)
        self.gcnLayer2 = GCNLayer(in_feats=3, out_feats=1, num_parts=num_parts)
    
    # def forward(self, graphStructure, subgraphFeature):
    def forward(self, subgraph, feat, norm, send_map, recv_map, rank, size):
        logits = self.gcnLayer1.forward(subgraph, feat, norm, send_map, recv_map, rank, size)
        # logits = self.gcnLayer2.forward(subgraph, logits, send_map, recv_map, rank, size)
        return logits
    
class GCNProtein(nn.Module):
    def __init__(self, in_feats, out_feats, part_size):
        super(GCNProtein, self).__init__()
        self.gcnLayer1 = GCNLayer(in_feats=1, out_feats=3, part_size=part_size, activation=F.relu)
        self.gcnLayer2 = GCNLayer(in_feats=3, out_feats=1, part_size=part_size, activation=F.relu)

        
    # def forward(self, graphStructure, subgraphFeature):
    def forward(self, g_strt, feat):
        logits = self.gcnLayer1.forward(g_strt, feat)
        logits = self.gcnLayer2.forward(g_strt, logits)
        return logits

class GCNPPI_SAGE(nn.Module):
    def __init__(self, in_feats, out_feats, part_size):
        super(GCNPPI_SAGE, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.activation = F.relu
        self.gcnLayer1 = GCNLayer(in_feats=in_feats, out_feats=2048, part_size=part_size)
        self.gcnLayer2 = GCNLayer(in_feats=2048, out_feats=2048, part_size=part_size)
        self.gcnLayer3 = GCNLayer(in_feats=2048, out_feats=2048, part_size=part_size)
        self.gcnLayer4 = GCNLayer(in_feats=2048, out_feats=2048, part_size=part_size)
        self.gcnLayer5 = GCNLayer(in_feats=2048, out_feats=2048, part_size=part_size)
        self.linear1 = nn.Linear(2048, 2048)
        self.linear2 = nn.Linear(2048, out_feats)
        self.layer_norm1 = nn.LayerNorm(2048, elementwise_affine=True)
        self.layer_norm2 = nn.LayerNorm(2048, elementwise_affine=True)
        self.layer_norm3 = nn.LayerNorm(2048, elementwise_affine=True)
        self.layer_norm4 = nn.LayerNorm(2048, elementwise_affine=True)
        self.layer_norm5 = nn.LayerNorm(2048, elementwise_affine=True)
        self.layer_norm6 = nn.LayerNorm(2048, elementwise_affine=True)
        
        register_hook_for_model_param(self.linear1.parameters())
        register_hook_for_model_param(self.linear2.parameters())
        register_hook_for_model_param(self.layer_norm1.parameters())
        register_hook_for_model_param(self.layer_norm2.parameters())
        register_hook_for_model_param(self.layer_norm3.parameters())
        register_hook_for_model_param(self.layer_norm4.parameters())
        register_hook_for_model_param(self.layer_norm5.parameters())
        register_hook_for_model_param(self.layer_norm6.parameters())

    # def forward(self, graphStructure, subgraphFeature):
    def forward(self, g_strt, feat):
        logits = self.gcnLayer1.forward(g_strt, feat)
        logits = self.layer_norm1(logits)
        logits = self.activation(logits)
        
        logits = self.gcnLayer2.forward(g_strt, logits)
        logits = self.layer_norm2(logits)
        logits = self.activation(logits)
        
        logits = self.gcnLayer3.forward(g_strt, logits)
        logits = self.layer_norm3(logits)
        logits = self.activation(logits)
        
        logits = self.gcnLayer4.forward(g_strt, logits)
        logits = self.layer_norm4(logits)
        logits = self.activation(logits)
        
        logits = self.gcnLayer5.forward(g_strt, logits)
        logits = self.layer_norm5(logits)
        logits = self.activation(logits)
        
        logits = self.linear1.forward(logits)
        logits = self.layer_norm1(logits)
        logits = self.activation(logits)
        
        logits = self.linear2.forward(logits)
        return logits