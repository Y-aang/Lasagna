import torch
import torch.distributed as dist
import torch.distributed.nn.functional as F
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

    # 每个 GPU 的发送数据大小不同
    send_sizes = [rank + 1 for _ in range(world_size)]  # 每个 GPU 发送的数据块大小
    recv_sizes = [(i + 1) for i in range(world_size)]   # 每个 GPU 接收的数据块大小

    # 创建发送和接收张量
    input_tensor = torch.cat([
        torch.full((size, 2), fill_value=rank, dtype=torch.float32, device=rank)
        for size in send_sizes
    ])
    output_tensor = torch.zeros(
        sum(recv_sizes), 2, dtype=torch.float32, device=rank
    )

    print(f"[Rank {rank}] Input tensor shape = {input_tensor.shape}")
    print(f"[Rank {rank}] Send sizes = {send_sizes}")
    print(f"[Rank {rank}] Recv sizes = {recv_sizes}")

    # 非均匀 All-to-All 通信
    F.all_to_all_single(
        output_tensor,
        input_tensor,
        recv_sizes,
        send_sizes
    )

    # 查看接收结果
    print(f"[Rank {rank}] Output tensor:\n{output_tensor}")

    # 销毁进程组
    dist.destroy_process_group()


def main():
    world_size = 4
    mp.spawn(worker, nprocs=world_size, args=(world_size,))


if __name__ == "__main__":
    main()
