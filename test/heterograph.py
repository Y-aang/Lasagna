import dgl
import torch
import dgl.function as fn


feat = torch.tensor([
    [1, 1, 1],
    [1, 2, 1],
    [1, 3, 1]
], dtype=torch.float)

u = torch.tensor([1, 2])
v = torch.tensor([0, 0])

g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
g.nodes['_U'].data['h'] = feat
g['_E'].update_all(fn.copy_u(u='h', out='m'),
                            fn.sum(msg='m', out='h'),
                            etype='_E')
print(g.nodes['_U'].data['h'])
print(g.nodes['_V'].data['h'])
