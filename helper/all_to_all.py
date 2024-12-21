import torch.distributed as dist
from typing import List
from torch import Tensor

# def all_to_all(recv_list: List[Tensor], send_list: List[Tensor]):
#     self_rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     part_size = len(recv_list)
#     offset = dist.get_rank() - dist.get_rank() % part_size
#     # send to next first
#     # recieve from last
#     dist.barrier()
#     for i in range(1, world_size):
#         send_pid = (self_rank + i) % world_size
#         # print(f'rsend from {self_rank} to {send_pid} {send_list[send_pid].shape}')
#         assert send_list[send_pid].is_cuda, f"send_list[{send_pid}] not on gpu"
#         send_req = dist.isend(send_list[send_pid], send_pid + offset)
        
#         recv_req = (self_rank - i) % world_size
#         # print(f'recieve {self_rank} from {recv_req} {recv_list[recv_req].shape}')
#         assert recv_list[recv_req].is_cuda, f"recv_list[{recv_req}] not on gpu"
#         recv = dist.irecv(recv_list[recv_req], recv_req + offset)
#         send_req.wait()
#         recv.wait()
#         # print(f'recieved {self_rank} from {recv_target}')
        
#         # dist.barrier()
#     dist.barrier()
#     return

def all_to_all(recv_list: List[Tensor], send_list: List[Tensor]):
    self_rank = dist.get_rank()
    world_size = dist.get_world_size()
    part_size = len(recv_list)
    offset = dist.get_rank() - dist.get_rank() % part_size
    dist.barrier()

    for i in range(1, world_size):
        send_pid = (self_rank + i) % world_size
        recv_pid = (self_rank - i) % world_size

        # 确保张量在 GPU 上
        assert send_list[send_pid].is_cuda, f"send_list[{send_pid}] not on GPU"
        assert recv_list[recv_pid].is_cuda, f"recv_list[{recv_pid}] not on GPU"

        # 构建通信操作列表
        op_list = []
        op_list.append(dist.P2POp(dist.isend, send_list[send_pid], send_pid + offset))
        op_list.append(dist.P2POp(dist.irecv, recv_list[recv_pid], recv_pid + offset))

        # 批量执行通信操作
        reqs = dist.batch_isend_irecv(op_list)
        for req in reqs:
            req.wait()  # 等待通信完成

    dist.barrier()
    return
