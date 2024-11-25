import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from module.layer import GCNLayer

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
    def forward(self, subgraph, feat, norm):
        logits = self.gcnLayer1.forward(subgraph, feat, norm)
        logits = self.gcnLayer2.forward(subgraph, logits, norm)
        return logits