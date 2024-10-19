import torch.distributed as dist
from typing import List
from torch import Tensor

def all_to_all(recv_list: List[Tensor], send_list: List[Tensor]):
    self_rank = dist.get_rank()
    world_size = dist.get_world_size()
    # send to next first
    # recieve from last
    dist.barrier()
    for i in range(1, world_size):
        target = (self_rank + i) % world_size
        # print(f'rsend from {self_rank} to {target} {send_list[target].shape}')
        feature = dist.isend(send_list[target], target)
        
        recv_target = (self_rank - i) % world_size
        # print(f'recieve {self_rank} from {recv_target} {recv_list[recv_target].shape}')
        recv = dist.irecv(recv_list[recv_target], recv_target)
        feature.wait()
        recv.wait()
        # print(f'recieved {self_rank} from {recv_target}')
        
        # dist.barrier()
    dist.barrier()
    return