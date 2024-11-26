from dgl.data import PPIDataset
from torch.utils.data import DataLoader

# 加载 PPI 数据集
train_dataset = PPIDataset(mode='train')
val_dataset = PPIDataset(mode='valid')
test_dataset = PPIDataset(mode='test')

