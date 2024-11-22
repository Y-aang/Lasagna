import dgl
from dgl.data import RedditDataset, YelpDataset, TUDataset
from dgl.partition import metis_partition
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from multiprocessing import Barrier
import os
from tqdm import tqdm
import copy
import pickle


class DevDataset(Dataset):
    def __init__(self, datasetName, datasetPath, nodeNumber=None, ):
        self.rank = dist.get_rank()
        if datasetName == 'reddit':
            dataset = RedditDataset(raw_dir=datasetPath)
        elif datasetName == 'yelp':
            dataset = YelpDataset(raw_dir=datasetPath)
        elif datasetName == 'proteins':
            dataset = TUDataset(name='PROTEINS', raw_dir=datasetPath)
            dataset = Subset(dataset, range(10))
        self.length = 0
        self.savePath = "./dataset"
        os.makedirs(self.savePath, exist_ok=True)
            
        meta_path = os.path.join(self.savePath, f'meta_data.pkl')
        if self.rank == 0:
            self.__group_and_partition(dataset)
            
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(meta_path, 'wb') as f:
                pickle.dump(self.length, f)
        else:
            print(dist.get_rank(), 'waiting...')
        dist.barrier()
        
        with open(meta_path, 'rb') as f:
            self.length = pickle.load(f)
        print(dist.get_rank(), 'dataset processing finished...')
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        rank = dist.get_rank()
        graph_save_path = os.path.join(self.savePath, f'graph_{index}')
        
        part_path = os.path.join(graph_save_path, f'part_{rank}.bin')
        parts, _ = dgl.load_graphs(part_path)
        part = parts[0]
        
        send_map_path = os.path.join(graph_save_path, f'send_map_{rank}.pt')
        recv_map_path = os.path.join(graph_save_path, f'recv_map_{rank}.pt')
        send_map = torch.load(send_map_path)
        recv_map = torch.load(recv_map_path)

        g_list_path = os.path.join(graph_save_path, f'g_list_{rank}.bin')
        g_structures, _ = dgl.load_graphs(g_list_path)
        g_structure = g_structures[0]

        return part, send_map, recv_map, g_structure

    def __group_and_partition(self, dataset):
        print("process 0 processing data...")

        k = 4

        for idx, data in enumerate(dataset):
            graph = self.__add_self_loop(data)
            self.__add_norm(graph)
            
            if graph.is_homogeneous:
                try:
                    self.__prepare_feat_tag(graph)
                    parts, g_list, send_map, recv_map= self.__process_graph(graph)
                    
                    graph_save_path = os.path.join(self.savePath, f'graph_{idx}')
                    os.makedirs(graph_save_path, exist_ok=True)
                    self.__save_graph(parts, g_list, send_map, recv_map, graph_save_path, k=4)
                    self.length += 1
                    print(f"Partitioning successed for graph_{idx}")
                except Exception as e:
                    print(f"Partitioning failed for graph_{idx} with {graph.num_nodes()} nodes: {e}")
            else:
                print(f"Graph {idx} is not homogeneous, skipping.")

    def __add_self_loop(self, data):
        graph = data[0]
        src, dst = graph.edges()
        src, dst = graph.edges()
        has_self_loops = (src == dst).all()
        if not has_self_loops:
            graph = dgl.add_self_loop(graph)
        return graph
    
    def __add_norm(self, graph):
        in_degrees = graph.in_degrees().float()
        norm = torch.pow(in_degrees, -0.5)
        norm[torch.isinf(norm)] = 0
        graph.ndata['norm'] = norm.unsqueeze(-1)

    def __prepare_feat_tag(self, graph):
        graph.ndata['feat'] = copy.deepcopy(graph.ndata['node_attr'])
        graph.ndata['tag'] = copy.deepcopy(graph.ndata['node_labels'])
    
    def __process_graph(self, graph):     # one graph prepare process
        # split the graph in to 4 parts using metis_partition
        
        num_parts = 4
        num_nodes = graph.num_nodes()
        parts = metis_partition(graph, k=num_parts)
        global_to_local_maps = {}
        # get partition number for each node through part_id
        node_part = torch.empty(num_nodes, dtype=torch.int64)
        # process distribution feature, node ranking and global_to_local_maps
        for part_id, subgraph in parts.items():
            global_node_ids = subgraph.ndata['_ID']
            subgraph.ndata['h'] = graph.ndata['feat'][global_node_ids].to(torch.float32)
            subgraph.ndata['tag'] = graph.ndata['tag'][global_node_ids].to(torch.float32)
            subgraph.ndata['norm'] = graph.ndata['norm'][global_node_ids].to(torch.float32)
            global_to_local = {global_id.item(): local_id for local_id, global_id in enumerate(global_node_ids)}
            global_to_local_maps[part_id] = global_to_local
            
        for part_id, subgraph in parts.items():
            node_part[subgraph.ndata['_ID']] = part_id  # 将子图的节点编号映射到原始图的节点

        # 第3步：记录每个子图的send和recv信息
        send_map = {i: {} for i in range(num_parts)}  # 记录每个子图要发送的目标子图
        recv_map = {i: {} for i in range(num_parts)}  # 记录每个子图要接收的源子图
        
        # 获取需要跨分区传递的边 (跨分区的节点)
        for u, v in zip(graph.edges()[0], graph.edges()[1]):
            part_u = node_part[u].item()  # u的分区
            part_v = node_part[v].item()  # v的分区
            if part_u != part_v:
                # u 需要发送特征给 v 的子图
                if part_v not in send_map[part_u]:
                    send_map[part_u][part_v] = []
                if u.item() not in send_map[part_u][part_v]:
                    send_map[part_u][part_v].append(u.item())
                # v 需要从 u 的子图接收特征
                if part_u not in recv_map[part_v]:
                    recv_map[part_v][part_u] = []
                if u.item() not in recv_map[part_v][part_u]:
                    recv_map[part_v][part_u].append(u.item())

        # 将send_map和recv_map转换为张量列表，方便后续处理
        for part in send_map:
            for target_part in send_map[part]:
                send_map[part][target_part] = torch.tensor(send_map[part][target_part])

        for part in recv_map:
            for source_part in recv_map[part]:
                recv_map[part][source_part] = torch.tensor(recv_map[part][source_part])

        # 更新global_to_local_maps
        global_to_local_maps = self.__update_global_to_local_maps(global_to_local_maps, recv_map)

        # 获取local的recv和send map用于作为dataset的处理后输出
        local_send_map = self.__convert_map_to_local(send_map, global_to_local_maps)
        local_recv_map = self.__convert_map_to_local(recv_map, global_to_local_maps)

        g_list = self.__construct_graph(graph, parts, send_map, recv_map, global_to_local_maps, node_part)
        
        return parts, g_list, local_send_map, local_recv_map
        
        
    def __save_graph(self, parts, g_list, send_map, recv_map, graph_save_path, k=4):
        for i in range(k):
            dgl.save_graphs(os.path.join(graph_save_path, f'part_{i}.bin'), [parts[i]])
            torch.save(send_map, os.path.join(graph_save_path, f'send_map_{i}.pt'))
            torch.save(recv_map, os.path.join(graph_save_path, f'recv_map_{i}.pt'))
            dgl.save_graphs(os.path.join(graph_save_path, f'g_list_{i}.bin'), [g_list[i]])
        
    def __update_global_to_local_maps(self, global_to_local_maps, recv_map):
        for rank, sub_map in recv_map.items():
            # Gather all unique nodes in the current rank's sub_map
            nodes = set()
            for target_rank, node_list in sub_map.items():
                nodes.update(node_list)

            # Sort the nodes to maintain a consistent order
            nodes = sorted(nodes)

            # Find the next available local index for the current rank
            current_max_index = max(global_to_local_maps[rank].values(), default=-1)
            next_index = current_max_index + 1

            # Add new nodes to the global_to_local_maps for the current rank
            for node in nodes:
                node = node.item()
                if node not in global_to_local_maps[rank]:
                    global_to_local_maps[rank][node] = next_index
                    next_index += 1
        return global_to_local_maps

    def __convert_map_to_local(self, recv_map, global_to_local_maps):
        # Create a deep copy of recv_map to avoid modifying the original one
        local_recv_map = copy.deepcopy(recv_map)
        
        for rank, sub_map in local_recv_map.items():
            # Iterate over the target ranks and node lists in the current rank's sub-map
            for target_rank, nodes in sub_map.items():
                # Update the nodes using global_to_local_maps
                local_recv_map[rank][target_rank] = [global_to_local_maps[rank].get(node.item()) for node in nodes]

        return local_recv_map

    def __construct_graph(self, graph, parts, send_map, recv_map, global_to_local_maps, node_part):
        u_subs, v_subs = [], []
        g_list = []
        for part_id, subgraph in parts.items():
            u, v = subgraph.edges()
            u_subs.append(u.tolist())
            v_subs.append(v.tolist())
        
        for u, v in zip(graph.edges()[0], graph.edges()[1]):
            part_u = node_part[u].item()
            part_v = node_part[v].item()
            
            if part_u != part_v:
                u_subs[part_v].append(global_to_local_maps[part_v][u.item()])
                v_subs[part_v].append(global_to_local_maps[part_v][v.item()])
                
        for part_id, subgraph in parts.items():
            num_nodes = subgraph.num_nodes()
            g = dgl.heterograph({('_U', '_E', '_V'): (u_subs[part_id], v_subs[part_id])})
            if g.num_nodes('_U') < num_nodes:
                g.add_nodes(num_nodes - g.num_nodes('_U'), ntype='_U')
            if g.num_nodes('_V') < num_nodes:
                g.add_nodes(num_nodes - g.num_nodes('_V'), ntype='_V')
            g_list.append(g)
        
        return g_list

    
def custom_collate_fn(batch):
    assert len(batch) == 1
    return batch[0]





# class LasagnaDataset:
    
#     def __init__(self, datasetName, datasetPath, nodeNumber=None, ):
#         self.rank = dist.get_rank()
#         if datasetName == 'reddit':
#             dataset = RedditDataset(raw_dir=datasetPath)
#         elif datasetName == 'yelp':
#             dataset = YelpDataset(raw_dir=datasetPath)
#         elif datasetName == 'proteins':
#             dataset = TUDataset(name='PROTEINS', raw_dir=datasetPath) 
            
#         if self.rank == 0:
#             self.group_and_partition(dataset, savePath="./dataset")
#             dist.barrier()
#         else:
#             print(dist.get_rank(), 'waiting...')
#             dist.barrier()
            
#         print(dist.get_rank(), 'processing...')
        
#     def __len__(self):
#         pass
    
#     def __getitem__(self, index):
#         pass
    
#     def group_and_partition(self, dataset, savePath):
#         print("process 0 processing data...")

#         k = 4
#         os.makedirs(savePath, exist_ok=True)

#         for idx, data in enumerate(dataset):
#             print("processing No. ", idx)
#             graph = data[0]
#             if graph.is_homogeneous:
#                 try:
#                     partitions = metis_partition(graph, k)
                    
#                     graph_savePath = os.path.join(savePath, f'graph_{idx}')
#                     os.makedirs(graph_savePath, exist_ok=True)
                    
#                     for i in range(k):
#                         part = partitions[i]
#                         dgl.save_graphs(os.path.join(graph_savePath, f'part_{i}.bin'), [part])
#                 except Exception as e:
#                     print(f"Error partitioning graph {idx}: {e}")
#             else:
#                 print(f"Graph {idx} is not homogeneous, skipping.")