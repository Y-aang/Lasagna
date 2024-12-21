import os
import torch
import torch.distributed as dist
from datetime import timedelta

def run(rank, size):
    # 每个 rank 绑定到对应的 GPU
    torch.cuda.set_device(rank)
    
    # 初始化张量
    if rank == 0:  # 发送方
        send_tensor = torch.ones(10, device=f'cuda:{rank}') * (rank + 1)  # 张量值为 rank + 1
        recv_tensor = torch.empty(10, device=f'cuda:{rank}')  # 接收缓冲区（未使用）
        print(f"Rank {rank} initialized send_tensor: {send_tensor}")
        # 异步发送
        send_req = dist.isend(tensor=send_tensor, dst=1)
        send_req.wait()
        print(f"Rank {rank} finished sending.")
    elif rank == 1:  # 接收方
        send_tensor = torch.empty(10, device=f'cuda:{rank}')  # 发送缓冲区（未使用）
        recv_tensor = torch.empty(10, device=f'cuda:{rank}')  # 接收缓冲区
        # 异步接收
        recv_req = dist.irecv(tensor=recv_tensor, src=0)
        recv_req.wait()
        print(f"Rank {rank} received tensor: {recv_tensor}")

def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29518'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=size, init_method='env://', timeout=timedelta(minutes=1.5))
    fn(rank, size)
    dist.destroy_process_group()

if __name__ == "__main__":
    size = 2  # 两个进程
    torch.multiprocessing.set_start_method('spawn')
    processes = []
    for rank in range(size):
        p = torch.multiprocessing.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
