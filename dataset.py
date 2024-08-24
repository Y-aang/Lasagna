import dgl
from dgl.data import RedditDataset, YelpDataset, TUDataset
import torch.distributed as dist
from dgl.partition import metis_partition
import os
from tqdm import tqdm

class LasagnaDataset:
    
    def __init__(self, datasetName, datasetPath, nodeNumber=None, ):
        self.rank = dist.get_rank()
        if datasetName == 'reddit':
            dataset = RedditDataset(raw_dir=datasetPath)
        elif datasetName == 'yelp':
            dataset = YelpDataset(raw_dir=datasetPath)
        elif datasetName == 'proteins':
            dataset = TUDataset(name='PROTEINS', raw_dir=datasetPath) 
            
        if self.rank == 0:
            self.group_and_partition(dataset, savePath="./dataset")
            dist.barrier()
        else:
            print(dist.get_rank(), 'waiting...')
            dist.barrier()
            
        print(dist.get_rank(), 'processing...')
        
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    def group_and_partition(self, dataset, savePath):
        print("process 0 processing data...")

        k = 4
        os.makedirs(savePath, exist_ok=True)

        for idx, data in enumerate(dataset):
            print("processing No. ", idx)
            graph = data[0]
            if graph.is_homogeneous:
                try:
                    partitions = metis_partition(graph, k)
                    
                    graph_savePath = os.path.join(savePath, f'graph_{idx}')
                    os.makedirs(graph_savePath, exist_ok=True)
                    
                    for i in range(k):
                        part = partitions[i]
                        dgl.save_graphs(os.path.join(graph_savePath, f'part_{i}.bin'), [part])
                except Exception as e:
                    print(f"Error partitioning graph {idx}: {e}")
            else:
                print(f"Graph {idx} is not homogeneous, skipping.")

    