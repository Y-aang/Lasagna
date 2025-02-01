import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv, GraphConv
import dgl


def generate_hs_init(G, hs, no_dim):
    max_sim = 0
    if 'batch' not in G.ndata:
        batch_size = 1
    else:
        batch_size = G.ndata['batch'].max().item() + 1
    for batch_idx in range(batch_size):
        if 'batch' not in G.ndata:
            pi_mask = (G.ndata['forward_level'] == 0)
        else:
            pi_mask = (G.ndata['batch'] == batch_idx) & (G.ndata['forward_level'] == 0)
        pi_node = G.ndata[dgl.NID][pi_mask]
        pi_vec, batch_max_sim = generate_orthogonal_vectors(len(pi_node), no_dim)
        if batch_max_sim > max_sim:
            max_sim = batch_max_sim
        hs[pi_node] = torch.tensor(pi_vec, dtype=torch.float)
    return hs, max_sim

class MLPGateDGL_bad(nn.Module):
    def __init__(self, args):
        super(MLPGateDGL, self).__init__()

        # Dimensions
        self.dim_node_feature = args['dim_node_feature']
        self.dim_hidden = args['dim_hidden']
        self.num_rounds = args['num_rounds']

        # Aggregation functions
        self.aggr_and_strc = GraphConv(self.dim_hidden, self.dim_hidden, activation=F.relu, allow_zero_in_degree=True)
        self.aggr_and_func = GraphConv(self.dim_hidden * 2, self.dim_hidden, activation=F.relu, allow_zero_in_degree=True)
        self.aggr_not_strc = GraphConv(self.dim_hidden, self.dim_hidden, activation=F.relu, allow_zero_in_degree=True)
        self.aggr_not_func = GraphConv(self.dim_hidden, self.dim_hidden, activation=F.relu, allow_zero_in_degree=True)
        self.aggr_or_strc = GraphConv(self.dim_hidden, self.dim_hidden, activation=F.relu, allow_zero_in_degree=True)
        self.aggr_or_func = GraphConv(self.dim_hidden * 2, self.dim_hidden, activation=F.relu, allow_zero_in_degree=True)

        # Update functions (GRU)
        self.update_and_strc = nn.GRU(self.dim_hidden, self.dim_hidden, batch_first=True)
        self.update_and_func = nn.GRU(self.dim_hidden, self.dim_hidden, batch_first=True)
        self.update_not_strc = nn.GRU(self.dim_hidden, self.dim_hidden, batch_first=True)
        self.update_not_func = nn.GRU(self.dim_hidden, self.dim_hidden, batch_first=True)
        self.update_or_strc = nn.GRU(self.dim_hidden, self.dim_hidden, batch_first=True)  # OR 门结构更新
        self.update_or_func = nn.GRU(self.dim_hidden, self.dim_hidden, batch_first=True)  # OR 门功能更新

        # Readout layers
        self.readout_prob = nn.Linear(self.dim_hidden, 1)
        self.readout_rc = nn.Linear(self.dim_hidden * 2, 1)

    def forward(self, g, rc_pair_index):
        # g = dgl.add_self_loop(g)  # Add self-loops to avoid 0-degree nodes
        num_nodes = g.num_nodes()
        num_layers_f = int(g.ndata['forward_level'].max().item()) + 1

        # Initialize hidden states
        hs = torch.ones(num_nodes, self.dim_hidden).to(g.device)  # Structure hidden state
        hf = torch.ones(num_nodes, self.dim_hidden).to(g.device)  # Function hidden state

        # Multi-round recursive updates
        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                # Mask nodes by level
                mask = g.ndata['forward_level'] == level
                sub_g = g.subgraph(mask)

                # AND Gate updates
                and_mask = sub_g.ndata['gate'] == 1
                if and_mask.sum() > 0:
                    sub_and_g = sub_g.subgraph(and_mask)

                    # Structure updates
                    msg = self.aggr_and_strc(sub_and_g, sub_and_g.ndata['x'])
                    and_msg = F.relu(msg)
                    and_hs = hs[sub_and_g.ndata[dgl.NID]]

                    # Reshape for GRU
                    and_msg = and_msg.unsqueeze(1)
                    and_hs = and_hs.unsqueeze(0)

                    # Call GRU
                    _, updated_hs = self.update_and_strc(and_msg, and_hs)
                    hs[sub_and_g.ndata[dgl.NID]] = updated_hs.squeeze(0)

                    # Function updates
                    concatenated_features = torch.cat([sub_and_g.ndata['x'], and_hs.squeeze(0)], dim=1)
                    msg = self.aggr_and_func(sub_and_g, concatenated_features)
                    and_msg = F.relu(msg)
                    and_hf = hf[sub_and_g.ndata[dgl.NID]]

                    # Reshape for GRU
                    and_msg = and_msg.unsqueeze(1)
                    and_hf = and_hf.unsqueeze(0)

                    # Call GRU
                    _, updated_hf = self.update_and_func(and_msg, and_hf)
                    hf[sub_and_g.ndata[dgl.NID]] = updated_hf.squeeze(0)

                # OR Gate updates
                or_mask = sub_g.ndata['gate'] == 2
                if or_mask.sum() > 0:
                    sub_or_g = sub_g.subgraph(or_mask)

                    # Structure updates
                    msg = self.aggr_or_strc(sub_or_g, sub_or_g.ndata['x'])
                    or_msg = F.relu(msg)
                    or_hs = hs[sub_or_g.ndata[dgl.NID]]

                    # Reshape for GRU
                    or_msg = or_msg.unsqueeze(1)
                    or_hs = or_hs.unsqueeze(0)

                    # Call GRU
                    _, updated_hs = self.update_or_strc(or_msg, or_hs)
                    hs[sub_or_g.ndata[dgl.NID]] = updated_hs.squeeze(0)

                    # Function updates
                    concatenated_features = torch.cat([sub_or_g.ndata['x'], or_hs.squeeze(0)], dim=1)
                    msg = self.aggr_or_func(sub_or_g, concatenated_features)
                    or_msg = F.relu(msg)
                    or_hf = hf[sub_or_g.ndata[dgl.NID]]

                    # Reshape for GRU
                    or_msg = or_msg.unsqueeze(1)
                    or_hf = or_hf.unsqueeze(0)

                    # Call GRU
                    _, updated_hf = self.update_or_func(or_msg, or_hf)
                    hf[sub_or_g.ndata[dgl.NID]] = updated_hf.squeeze(0)

        # Readout layer
        prob = torch.sigmoid(self.readout_prob(hf))
        rc_emb = torch.cat([hs[rc_pair_index[0]], hs[rc_pair_index[1]]], dim=1)
        is_rc = torch.sigmoid(self.readout_rc(rc_emb))

        return hs, hf, prob, is_rc


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import dgl
from mlp import MLP


class MLPGateDGL(nn.Module):
    def __init__(self, args):
        super(MLPGateDGL, self).__init__()

        self.dim_node_feature = args['dim_node_feature']
        self.dim_hidden = args['dim_hidden']
        self.dim_mlp = args['dim_mlp']
        self.num_rounds = args['num_rounds']
        self.device = args['device']

        # Aggregation functions
        # self.aggr_and_strc = GraphConv(self.dim_hidden, self.dim_hidden, activation=F.relu, allow_zero_in_degree=True)
        self.aggr_and_strc = MLP(self.dim_hidden*1, self.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
        self.aggr_and_func = MLP(self.dim_hidden*2, self.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
        self.aggr_not_strc = MLP(self.dim_hidden*1, self.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
        self.aggr_not_func = MLP(self.dim_hidden*2, self.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')

        # Update functions (GRU)
        self.update_and_strc = nn.GRU(self.dim_hidden, self.dim_hidden, batch_first=True)
        self.update_and_func = nn.GRU(self.dim_hidden, self.dim_hidden, batch_first=True)
        self.update_not_strc = nn.GRU(self.dim_hidden, self.dim_hidden, batch_first=True)
        self.update_not_func = nn.GRU(self.dim_hidden, self.dim_hidden, batch_first=True)

        # Readout layers
        self.readout_prob = MLP(self.dim_hidden, self.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.readout_rc = MLP(self.dim_hidden * 2, self.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', sigmoid=True)

        # Embedding for function hidden state
        self.one = torch.ones(1, device=self.device)
        self.hf_emd_int = nn.Linear(1, self.dim_hidden)
        self.one.requires_grad = False


    def forward(self, g, rc_pair_index):
        num_nodes = g.num_nodes()
        num_layers_f = int(g.ndata['forward_level'].max().item()) + 1

        # Initialize hidden states
        hs = torch.ones(num_nodes, self.dim_hidden, device=self.device)  # Structure hidden state
        # hf = self.hf_emd_int(self.one).repeat(num_nodes, 1)  # Function hidden state
        hf = torch.ones(num_nodes, self.dim_hidden, device=self.device)

        # Multi-round recursive updates
        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                # Mask nodes by level
                mask = g.ndata['forward_level'] == level
                sub_g = g.subgraph(mask)

                # AND Gate updates
                and_mask = sub_g.ndata['gate'] == 1
                if and_mask.sum() > 0:
                    self._update_gate(sub_g, hs, hf, self.aggr_and_strc, self.update_and_strc, 
                                      self.aggr_and_func, self.update_and_func, and_mask)

                # NOT Gate updates
                not_mask = sub_g.ndata['gate'] == 2
                if not_mask.sum() > 0:
                    self._update_gate(sub_g, hs, hf, self.aggr_not_strc, self.update_not_strc, 
                                      self.aggr_not_func, self.update_not_func, not_mask)

        # Readout
        prob = (self.readout_prob(hf)) #加torch.sigmoid 不报错
        rc_emb = torch.cat([hs[rc_pair_index[0]], hs[rc_pair_index[1]]], dim=1)
        is_rc = (self.readout_rc(rc_emb))

        return hs, hf, prob, is_rc

    def _update_gate(self, sub_g, hs, hf, aggr_strc, update_strc, aggr_func, update_func, mask):
        sub_gate_g = sub_g.subgraph(mask)

        # Structure updates
        msg = aggr_strc(sub_gate_g.ndata['x'])
        strc_msg = msg
        strc_hs = hs[sub_gate_g.ndata[dgl.NID]]

        # Reshape for GRU
        strc_msg = strc_msg.unsqueeze(1)  # (num_nodes, 1, dim_hidden)
        strc_hs = strc_hs.unsqueeze(0)    # (1, num_nodes, dim_hidden)

        # Update GRU
        _, updated_hs = update_strc(strc_msg, strc_hs)
        hs[sub_gate_g.ndata[dgl.NID]] = updated_hs.squeeze(0)

        # Function updates
        concatenated_features = torch.cat([sub_gate_g.ndata['x'], hf[sub_gate_g.ndata[dgl.NID]]], dim=1)
        msg = aggr_func(concatenated_features)
        func_msg = msg
        func_hf = hf[sub_gate_g.ndata[dgl.NID]]

        # Reshape for GRU
        func_msg = func_msg.unsqueeze(1)  # (num_nodes, 1, dim_hidden)
        func_hf = func_hf.unsqueeze(0)    # (1, num_nodes, dim_hidden)

        # Update GRU
        _, updated_hf = update_func(func_msg, func_hf)
        hf[sub_gate_g.ndata[dgl.NID]] = updated_hf.squeeze(0)
