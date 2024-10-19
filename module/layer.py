from torch import nn
import torch
import torch.distributed as dist
import torch.nn.init as init
import dgl.function as fn
from helper.all_to_all import all_to_all

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_parts):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.num_parts = num_parts
        
        init.constant_(self.linear.weight, 1)
        init.constant_(self.linear.bias, 1)
    
    # def forward(self, graphStructure, subgraphFeature):
    def forward(self, subgraph, feat, send_map, recv_map, rank, size):
        dist.barrier()
        print(f"Rank {rank}: 进入消息传递阶段")
        ops = []
        # feat.register_hook(communicate_grad)
        send_list = [torch.tensor([0.0])] * size
        
        for part_v in send_map[rank]:
            send_feat = feat[send_map[rank][part_v]].clone()
            print(f"Rank {rank}: 发送特征到 {part_v}, send_feat.shape={send_feat.shape}")
            send_list[part_v] = send_feat
        output = [torch.empty(1)] * size
        for part_v in recv_map[rank]:
            recv_feat = torch.empty((len(recv_map[rank][part_v]), feat.shape[1])) # num_nodes_to_receive， feature_dim
            output[part_v] = recv_feat
        print('Go Send')
        all_to_all(output, send_list)
        # dist.barrier()
        print(f"Rank {rank}: 进入消息接受阶段")
        
        feat_expand = torch.empty(subgraph.num_nodes('_U') - feat.shape[0], feat.shape[1])
        feat = torch.cat((feat, feat_expand), dim=0)
        for part_v in recv_map[rank]:
            recv_feat = output[part_v]
            feat[recv_map[rank][part_v]] = recv_feat
            print(f"Rank {rank}: 接收特征来自 {part_v}, recv_feat.shape={recv_feat.shape}")
        print(f"Rank {rank}: Finish")
        dist.barrier()

        subgraph.nodes['_U'].data['h'] = feat
        subgraph.update_all(fn.copy_u(u='h', out='m'),
                                 fn.sum(msg='m', out='h'))
        h = subgraph.nodes['_V'].data['h']
        h = self.linear(h)
        return h
    
    
def communicate_grad(grad: torch.Tensor):
    # start to send the grad and retrieve
    self_rank = dist.get_rank()
    world_size = dist.get_world_size()
    recv_list = [torch.tensor([0]) if i == self_rank else torch.empty_like(feat[send_map[self_rank][i]]) for i in range(world_size)]
    send_list = [torch.tensor([0]) if i == self_rank else grad[recv_map[self_rank][i]] for i in range(world_size)]
    
    all_to_all(recv_list, send_list)
    
    for i in range(world_size):
        if i == self_rank:
            continue
        grad[send_map[self_rank][i]] += recv_list[i]
    return grad
