import torch
from torch_geometric.data import Data

# Path to your saved `data.pt`
data_pt_path = "/data/yangshen/template/DeepGate2/data/train/inmemory_norc/data.pt"

# Load (data, slices) tuple
data, slices = torch.load(data_pt_path)

# Reconstruct the very first graph from the slices.
# We'll build a new torch_geometric.data.Data object manually:
data_0 = Data()
for key in data.__dict__:
    item = data[key]
    # If it's a tensor, slice the correct range
    if torch.is_tensor(item):
        start = slices[key][0]
        end = slices[key][1]
        data_0[key] = item[start:end]
    else:
        # Otherwise just assign it directly
        data_0[key] = item

# Now print the first data example
print("=== First Graph (data_0) ===")
print(data_0)
print("Node features (x):", data_0.x)
print("Edge index (edge_index):", data_0.edge_index)
print("Other attributes:", data_0.keys)
