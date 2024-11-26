import torch.distributed as dist
from typing import List
from torch import Tensor

def all_to_all(recv_list: List[Tensor], send_list: List[Tensor]):
    self_rank = dist.get_rank()
    world_size = dist.get_world_size()
    part_size = len(recv_list)
    offset = dist.get_rank() - dist.get_rank() % part_size
    # send to next first
    # recieve from last
    dist.barrier()
    for i in range(1, world_size):
        send_pid = (self_rank + i) % world_size
        # print(f'rsend from {self_rank} to {target} {send_list[target].shape}')
        assert send_list[send_pid].is_cuda, f"send_list[{send_pid}] not on gpu"
        send_req = dist.isend(send_list[send_pid], send_pid + offset)
        
        recv_req = (self_rank - i) % world_size
        # print(f'recieve {self_rank} from {recv_target} {recv_list[recv_target].shape}')
        assert recv_list[recv_req].is_cuda, f"recv_list[{recv_req}] not on gpu"
        recv = dist.irecv(recv_list[recv_req], recv_req + offset)
        send_req.wait()
        recv.wait()
        # print(f'recieved {self_rank} from {recv_target}')
        
        # dist.barrier()
    dist.barrier()
    return