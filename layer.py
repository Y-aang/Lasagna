from torch import nn
import torch
import torch.distributed as dist
import torch.nn.init as init
import dgl.function as fn
from all_to_all import all_to_all

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
            # dist.isend(tensor=send_feat, dst=part_v)
            send_list[part_v] = send_feat
            # print("send:", dist.isend(tensor=send_feat, dst=part_v))
        output = [torch.empty(1)] * size
        for part_v in recv_map[rank]:
            recv_feat = torch.empty_like(feat[recv_map[rank][part_v]])
            output[part_v] = recv_feat
        # dist.all_to_all(output, send_list)
        print('Go Send')
        all_to_all(output, send_list)
        # dist.barrier()
        print(f"Rank {rank}: 进入消息接受阶段")
        for part_v in recv_map[rank]:
            recv_feat = output[part_v]
            # dist.irecv(tensor=recv_feat, src=part_v).wait()
            feat[recv_map[rank][part_v]] = recv_feat
            print(f"Rank {rank}: 接收特征来自 {part_v}, recv_feat.shape={recv_feat.shape}")
        print(f"Rank {rank}: Finish")
        dist.barrier()

        subgraph.ndata['h'] = feat
        subgraph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        h = subgraph.ndata.pop('h')
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
