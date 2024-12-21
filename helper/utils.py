import torch
import torch.distributed as dist
from helper.all_to_all import all_to_all

def parameter_hook():
    def communicate_grad(grad):
        # rank = dist.get_rank()
        # size = dist.get_world_size() 
        # send_list = [grad.clone() for _ in range(size)]
        # recv_list = [torch.zeros_like(grad) for _ in range(size)]
        # all_to_all(recv_list, send_list)
        
        # recv_list[rank] = grad
        # grad_sum = sum(recv_list)
        # return grad_sum
        # dist.all_reduce(grad, op=dist.ReduceOp.SUM, device='cuda')
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        return grad
    return communicate_grad

def register_hook_for_model_param(params):
    for param in params:
        if param.requires_grad:
            param.register_hook(parameter_hook())
            
def feat_hook(send_map, recv_map):
    def communicate_grad(grad):
        rank = dist.get_rank()
        size = dist.get_world_size() 
        send_list = [torch.tensor([0.0], device='cuda')] * size
        recv_list = [torch.tensor([0.0], device='cuda')] * size
        for src, feat_idx in recv_map[rank].items():
            send_list[src] = grad[feat_idx]
        for tgt, feat_idx in send_map[rank].items():
            recv_list[tgt] = torch.empty((len(feat_idx), grad.shape[1]), device='cuda')
        all_to_all(recv_list, send_list)
        for tgt, feat_idx in send_map[rank].items():
            grad[feat_idx] = grad[feat_idx] + recv_list[tgt]
            
        return grad
    return communicate_grad

def average_loss(loss, n_node):
    n_train = torch.tensor(n_node, dtype=torch.float)
    # dist.all_reduce(n_train, op=dist.ReduceOp.SUM, device='cuda')
    dist.all_reduce(n_train.to('cuda'), op=dist.ReduceOp.SUM)
    loss /= n_train


# def communicate_grad(grad: torch.Tensor):
#     # start to send the grad and retrieve
#     self_rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     recv_list = [torch.tensor([0]) if i == self_rank else torch.empty_like(feat[send_map[self_rank][i]]) for i in range(world_size)]
#     send_list = [torch.tensor([0]) if i == self_rank else grad[recv_map[self_rank][i]] for i in range(world_size)]
    
#     all_to_all(recv_list, send_list)
    
#     for i in range(world_size):
#         if i == self_rank:
#             continue
#         grad[send_map[self_rank][i]] += recv_list[i]
#     return grad
