import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank, world_size):
    # 初始化分布式
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

    # 分组：每两个 GPU 一组
    group_id = rank // 2
    partner_rank = rank + 1 if rank % 2 == 0 else rank - 1  # 组内的通信伙伴

    # 准备发送数据和接收缓冲区
    send_data = torch.full((rank + 1, group_id + 2), fill_value=rank, device='cuda')  # 发送数据 shape: (rank+1, group_id+2)
    recv_data = torch.empty((partner_rank + 1, group_id + 2), device='cuda')          # 接收缓冲区 shape: (partner_rank+1, group_id+2)

    # 构建通信操作列表
    op_list = []
    op_list.append(dist.P2POp(dist.isend, send_data, partner_rank))
    op_list.append(dist.P2POp(dist.irecv, recv_data, partner_rank))

    # 批量执行通信操作
    reqs = dist.batch_isend_irecv(op_list)
    for req in reqs:
        req.wait()  # 等待所有通信完成

    # 打印发送和接收的结果
    print(f"[Rank {rank}] Sent data shape: {send_data.shape}")
    print(f"[Rank {rank}] Received data shape: {recv_data.shape}")

    # 销毁进程组
    dist.destroy_process_group()


def main():
    world_size = 8  # 8 个 GPU，两两分组
    mp.spawn(worker, nprocs=world_size, args=(world_size,))


if __name__ == "__main__":
    main()
