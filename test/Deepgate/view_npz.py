import numpy as np

# 读取数据
graphs = np.load('/data/yangshen/template/DeepGate2/data/train/graphs.npz', allow_pickle=True)['circuits'].item()
labels = np.load('/data/yangshen/template/DeepGate2/data/train/labels.npz', allow_pickle=True)['labels'].item()

# 遍历电路
for idx, circuit_name in enumerate(graphs):
    if idx > 1:
        break
    print(f"电路名: {circuit_name}")
    
    # 获取节点特性和边关系
    x_data = graphs[circuit_name]['x']
    edge_index = graphs[circuit_name]['edge_index']
    print("节点特性：", x_data)
    print("边关系：", edge_index)
    
    # 获取标签数据
    tt_pair_index = labels[circuit_name]['tt_pair_index']
    tt_dis = labels[circuit_name]['tt_dis']
    prob = labels[circuit_name]['prob']
    min_tt_dis = labels[circuit_name]['min_tt_dis']
    print("节点对索引：", tt_pair_index)
    print("真值表距离：", tt_dis)
    print("节点概率：", prob)
    print("最小化距离：", min_tt_dis)
