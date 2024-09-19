from layer import GCNLayer

class myGCN():
    def __init__(self):
        self.gcnLayer1 = GCNLayer()
    
    def forward(self, graphStructure, subgraphFeature):
        logits = self.gcnLayer1(graphStructure, subgraphFeature)
        return logits