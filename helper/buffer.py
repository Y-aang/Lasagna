import torch.distributed as dist
from torch import Tensor

import torch.distributed.rpc as rpc
import torch.futures
import os
from typing import List
import asyncio
local_rank = os.environ["LOCAL_RANK"]
world_rank = os.environ["WORLD_RANK"]


class LocalFeatureStore():

    def __init__(self):
        self.feature_map = {}

    def update_node_feature_map(self, map: dict[int, Tensor], layer: int):
        self.feature_map[layer] = map

    def set_node_map(self, map: dict[int, int]):
        self.map = map

    async def lookup_features(self, index: List[int], layer=0):
        # get nodes's machine
        machine_nodes = {}
        for i in index:
            if i not in machine_nodes:
                machine_nodes[i] = []
            machine_nodes[i].append(self.map[i])

        res = {}
        features = []
        for k, v in machine_nodes.items():
            if k == world_rank:
                # nodes on current machine
                for i in index:
                    while i not in self.feature_map[layer]:
                        await asyncio.sleep(0.1)
                    res[i] = self.feature_map[layer][i]
            features.append((rpc.rpc_async(
                f'worker_{k}', self.lookup_features, args=(v, layer)), v))
        for f in features:
            fs = f[0].wait()
            for i in range(len(f[1])):
                res[f[1][i]] = fs[i]
        return [res[x] for x in index]
