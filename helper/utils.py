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


def evaluate(model, test_loader):
    true_pos = torch.tensor([0], device='cuda')
    true_neg = torch.tensor([0], device='cuda')
    false_pos = torch.tensor([0], device='cuda')
    false_neg = torch.tensor([0], device='cuda')
    for g_strt, feat, tag in test_loader:
        device = torch.device('cuda')
        g_strt = g_strt.to(device)
        feat = feat.to(device)
        tag = tag.to(device)
        logits = model.forward(g_strt, feat)
        y_true = tag.long()
        y_pred = (logits > 0).long()
        true_pos += (y_true * y_pred).sum()
        false_pos += ((1 - y_true) * y_pred).sum()
        false_neg += (y_true * (1 - y_pred)).sum()

    dist.all_reduce(true_pos, op=dist.ReduceOp.SUM)
    dist.all_reduce(true_neg, op=dist.ReduceOp.SUM)
    dist.all_reduce(false_pos, op=dist.ReduceOp.SUM)
    dist.all_reduce(false_neg, op=dist.ReduceOp.SUM)

    return (true_pos / (true_pos + 0.5 * (false_pos + false_neg))).item()


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
