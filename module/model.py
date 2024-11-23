import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from module.layer import GCNLayer
from helper.utils import register_hook_for_model

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
    def __init__(self, in_feats, out_feats, num_parts):
        super(GCNProtein, self).__init__()
        self.gcnLayer1 = GCNLayer(in_feats=1, out_feats=3, num_parts=num_parts, activation=F.relu)
        self.gcnLayer2 = GCNLayer(in_feats=3, out_feats=3, num_parts=num_parts, activation=F.relu)
        self.gcnLayer3 = GCNLayer(in_feats=3, out_feats=3, num_parts=num_parts, activation=F.relu)
        self.gcnLayer4 = GCNLayer(in_feats=3, out_feats=3, num_parts=num_parts, activation=F.relu)
        self.gcnLayer5 = GCNLayer(in_feats=3, out_feats=3, num_parts=num_parts, activation=F.relu)
        self.gcnLayer6 = GCNLayer(in_feats=3, out_feats=1, num_parts=num_parts, activation=F.relu)

        
        register_hook_for_model(self.gcnLayer1, dist.get_rank(), dist.get_world_size())
        register_hook_for_model(self.gcnLayer2, dist.get_rank(), dist.get_world_size())
        register_hook_for_model(self.gcnLayer3, dist.get_rank(), dist.get_world_size())
        register_hook_for_model(self.gcnLayer4, dist.get_rank(), dist.get_world_size())
        register_hook_for_model(self.gcnLayer5, dist.get_rank(), dist.get_world_size())
        register_hook_for_model(self.gcnLayer6, dist.get_rank(), dist.get_world_size())
        
    # def forward(self, graphStructure, subgraphFeature):
    def forward(self, subgraph, feat, norm, send_map, recv_map, rank, size):
        logits = self.gcnLayer1.forward(subgraph, feat, norm, send_map, recv_map, rank, size)
        logits = self.gcnLayer2.forward(subgraph, logits, norm, send_map, recv_map, rank, size)
        logits = self.gcnLayer3.forward(subgraph, logits, norm, send_map, recv_map, rank, size)
        logits = self.gcnLayer4.forward(subgraph, logits, norm, send_map, recv_map, rank, size)
        logits = self.gcnLayer5.forward(subgraph, logits, norm, send_map, recv_map, rank, size)
        logits = self.gcnLayer6.forward(subgraph, logits, norm, send_map, recv_map, rank, size)
        return logits