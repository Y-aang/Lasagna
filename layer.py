from torch import nn

class GCNLayer(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, graphStructure, subgraphFeature):
        send_targets = map_send_recv[rank]['send']
        recv_sources = map_send_recv[rank]['recv']

        torch.cuda.synchronize()

        for target_rank in send_targets:
            dist.send(tensor=subgraphFeature, dst=target_rank)

        received_data = []
        for source_rank in recv_sources:
            recv_tensor = torch.zeros_like(subgraphFeature)
            dist.recv(tensor=recv_tensor, src=source_rank)
            received_data.append(recv_tensor)

        

        # 返回汇总后的数据，可以根据需求处理received_data
        return torch.stack(received_data).sum(dim=0)