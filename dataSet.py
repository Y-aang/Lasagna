import dgl
from dgl.data import RedditDataset, YelpDataset
import torch.distributed as dist

class LasagnaDataset:
    
    def __init__(self, datasetName, datasetPath, nodeNumber=None, ):
        self.rank = dist.get_rank()
        if datasetName == 'reddit':
            dataset = RedditDataset(raw_dir=datasetPath)
        elif datasetName == 'yelp':
            dataset = YelpDataset(raw_dir=datasetPath)
            
        if self.rank == 0:
            self.group_and_partition(dataset)
            dist.barrier()
        else:
            print(dist.get_rank(), 'waiting...')
            dist.barrier()
            
        print(dist.get_rank(), 'processing...')
        
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    def group_and_partition(self, dataset):
        print("process 0 processing data...")
        # pairs = []
        # for i in range(0, len(dataset), 2):
        #     if i + 1 < len(graphs):
        #         pairs.append([graphs[i], graphs[i + 1]])
        #     else:
        #         pairs.append([graphs[i]])
        # return pairs
    