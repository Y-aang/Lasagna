import torch
import torch.distributed as dist
from helper.all_to_all import all_to_all

def parameter_hook(rank, size):
    def communicate_grad(grad):
        send_list = [grad.clone() for _ in range(dist.get_world_size())]
        recv_list = [torch.zeros_like(grad) for _ in range(dist.get_world_size())]
        all_to_all(recv_list, send_list)
        
        recv_list[rank] = grad
        grad_sum = sum(recv_list)
        return grad_sum
    return communicate_grad

def register_hook_for_model(model, rank, size):
    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(parameter_hook(rank, size))


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
