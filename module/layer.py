from torch import nn
import torch
import torch.distributed as dist
import torch.nn.init as init
import dgl.function as fn
from helper.all_to_all import all_to_all
from helper.utils import feat_hook

class GNNBase(nn.Module):
    def __init__(self):
        super(GNNBase, self).__init__()
    
    def distributed_comm(self, subgraph, feat, send_map, recv_map, rank, size):
        dist.barrier()
        # feat.register_hook(communicate_grad)
        send_list, recv_list = self.__prepare_comm_data(feat, send_map, recv_map, rank, size)
        all_to_all(recv_list, send_list)
        feat = self.__process_recv_data(subgraph, feat, recv_map, recv_list, rank)
        
        if feat.requires_grad:
            feat.register_hook(feat_hook(send_map, recv_map, rank, size))
        
        dist.barrier()
        return feat
    
    def __prepare_comm_data(self, feat, send_map, recv_map, rank, size):
        send_list = [torch.tensor([0.0])] * size
        for part_v in send_map[rank]:
            send_feat = feat[send_map[rank][part_v]].clone()
            send_list[part_v] = send_feat
        recv_list = [torch.empty(1)] * size
        for part_v in recv_map[rank]:
            recv_feat = torch.empty((len(recv_map[rank][part_v]), feat.shape[1])) # num_nodes_to_receive， feature_dim
            recv_list[part_v] = recv_feat
        return send_list, recv_list
    
    def __process_recv_data(self, subgraph, feat, recv_map, recv_list, rank):
        feat_expand = torch.empty(subgraph.num_nodes('_U') - feat.shape[0], feat.shape[1])
        feat = torch.cat((feat, feat_expand), dim=0)
        for part_v in recv_map[rank]:
            recv_feat = recv_list[part_v]
            feat[recv_map[rank][part_v]] = recv_feat
        return feat
            

class GCNLayer(GNNBase):
    def __init__(self, in_feats, out_feats, num_parts):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.num_parts = num_parts
        
        init.constant_(self.linear.weight, 1)
        init.constant_(self.linear.bias, 1)
    
    # def forward(self, graphStructure, subgraphFeature):
    def forward(self, subgraph, feat, send_map, recv_map, rank, size):
        feat = super().distributed_comm(subgraph, feat, send_map, recv_map, rank, size)
        subgraph.nodes['_U'].data['h'] = feat
        subgraph.update_all(fn.copy_u(u='h', out='m'),
                                 fn.sum(msg='m', out='h'))
        h = subgraph.nodes['_V'].data['h']
        h = self.linear(h)
        return h
