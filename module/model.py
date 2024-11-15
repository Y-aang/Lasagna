import torch.nn as nn
from module.layer import GCNLayer

class myGCN(nn.Module):
    def __init__(self, in_feats, out_feats, num_parts):
        super(myGCN, self).__init__()
        self.gcnLayer1 = GCNLayer(in_feats=3, out_feats=3, num_parts=num_parts)
        self.gcnLayer2 = GCNLayer(in_feats=3, out_feats=3, num_parts=num_parts)
    
    # def forward(self, graphStructure, subgraphFeature):
    def forward(self, subgraph, feat, send_map, recv_map, rank, size):
        logits = self.gcnLayer1(subgraph, feat, send_map, recv_map, rank, size)
        logits = self.gcnLayer2(subgraph, logits, send_map, recv_map, rank, size)
        return logits