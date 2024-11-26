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

class GCNPPI(nn.Module):
    def __init__(self, in_feats, out_feats, part_size):
        super(GCNPPI, self).__init__()
        self.gcnLayer1 = GCNLayer(in_feats=in_feats, out_feats=2048, part_size=part_size, activation=F.relu)
        self.gcnLayer2 = GCNLayer(in_feats=2048, out_feats=2048, part_size=part_size, activation=F.relu)
        self.linear1 = nn.Linear(2048, out_feats)
        
        init.constant_(self.linear1.weight, 1)
        init.constant_(self.linear1.bias, 1)
        register_hook_for_model_param(self.linear1.parameters())

    # def forward(self, graphStructure, subgraphFeature):
    def forward(self, g_strt, feat):
        logits = self.gcnLayer1.forward(g_strt, feat)
        logits = self.gcnLayer2.forward(g_strt, logits)
        logits = self.linear1.forward(logits)
        return logits