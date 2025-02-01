import torch
import dgl

# 如果需要在脚本中使用你自定义的模型，比如 MLPGateDGL，需要先导入
# from my_models import MLPGateDGL  # 假设你的模型定义在 my_models.py 中


def create_graph_and_data(args):
    """
    创建一个示例 graph，并附带 rc_pair_index, labels 等数据。
    """
    # 1. 创建 DGL Graph
    g = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]), num_nodes=5)
    
    # 2. 节点特征（这里仅做演示，全部置为 0）
    g.ndata['x'] = torch.zeros(5, args['dim_hidden'])

    # 3. 其他节点数据（门类型、前向层次）
    g.ndata['gate'] = torch.tensor([0, 1, 1, 0, 0])          # 0=PI, 1=AND, 2=NOT
    g.ndata['forward_level'] = torch.tensor([0, 1, 1, 0, 0]) # 任意示例值

    # 4. 示例 rc_pair_index
    rc_pair_index = torch.tensor([[0, 1], [2, 3]])

    # 5. 示例二分类标签
    labels = torch.randint(0, 2, (5,)).float()  # 0 或 1

    # 返回三个对象
    return g, rc_pair_index, labels


def main():
    # 设定超参数
    args = {
        'dim_node_feature': 32,
        'dim_hidden': 4,
        'dim_mlp': 8,
        'num_rounds': 1,
        'device': 'cuda',  # 你也可以指定 'cpu'
    }

    # ====== 1. 创建多个图及其相关数据并保存 ======
    data_list = []
    for i in range(5):
        g, rc_pair_index, labels = create_graph_and_data(args)
        data_list.append((g, rc_pair_index, labels))

    # 也可以在这里初始化并保存你的模型，如果有需要的话：
    # model = MLPGateDGL(args).to(args['device'])
    # for param in model.parameters():
    #     param.data.fill_(3.0)
    # data_list.append(model.state_dict())  # 比如将 model 的权重也放进来

    # 将所有数据打包保存到 .pt 文件
    torch.save(data_list, "./graphs_data.pt")
    print("已将 5 个图及相关数据保存到 'graphs_data.pt'")

    # ====== 2. 从 .pt 文件中读取并验证 ======
    loaded_data_list = torch.load("./graphs_data.pt")
    print("已从 'graphs_data.pt' 读取数据")

    # 如果当时也保存了模型的 state_dict，可以在这里恢复
    # *注意*：需要和保存时顺序对应，比如 data_list 最后一个是 model 的 state_dict
    # loaded_model_state_dict = loaded_data_list[-1]
    # model.load_state_dict(loaded_model_state_dict)
    # loaded_data_list = loaded_data_list[:-1]  # 拿掉最后的 state_dict

    # 依次验证每个图
    for i, (g, rc_pair_index, labels) in enumerate(loaded_data_list):
        print(f"\n=== 第 {i} 个 Graph ===")
        print("图的边：", g.edges())
        print("ndata['gate']：", g.ndata['gate'])
        print("ndata['forward_level']：", g.ndata['forward_level'])
        print("rc_pair_index：\n", rc_pair_index)
        print("labels：\n", labels)


if __name__ == "__main__":
    main()
