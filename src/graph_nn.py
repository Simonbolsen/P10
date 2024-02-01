import numpy as np
import file_util as fu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
import tensor_network_util as tnu

class EdgePredictionGNN(nn.Module):
    def __init__(self, hidden_size):
        super(EdgePredictionGNN, self).__init__()
        self.node_layer = torch.nn.Linear(1, hidden_size)
        self.conv1 = GCNConv(hidden_size, hidden_size)
        self.edge_layer = torch.nn.Linear(hidden_size, 1)

    def get_edge_features(self, x, edge_index):
        src, dst = edge_index
        return x[src] + x[dst]

    def forward(self, data):
        edge_index = data.edge_index
        x = torch.tensor([[len(s) for s in data.shape]], dtype=torch.float).transpose(0,1)

        x = self.node_layer(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.get_edge_features(x, edge_index)
        x = self.edge_layer(x)

        return x
    
def get_path_from_values(edge_index, edge_values):
    ei = edge_index.transpose(0,1)

    pairs = []
    part_of = {}

    for i, edge in enumerate(ei):
        e0 = edge[0].item()
        e1 = edge[1].item()
        pairs.append((e0, e1, edge_values[i].item()))
        part_of[e0] = e0
        part_of[e1] = e1

    pairs = sorted(pairs, key=lambda x:x[2])

    def get_current_tensor(i):
        p = part_of[i]
        return p if p == i else get_current_tensor(p)

    path = []

    while len(pairs) > 0:
        pair = pairs.pop()
        left = get_current_tensor(pair[0])
        right = get_current_tensor(pair[1])

        if left != right:
            path.append((left, right))
            part_of[left] = right

    return path

if __name__ == "__main__":
    graph = fu.load_nx_graph("C:\\Users\\simon\\Documents\\GitHub\\P10\\graphs\\graph_dj_q5.gml")
    data = from_networkx(graph)

    hidden_size = 32

    model = EdgePredictionGNN(hidden_size)

    edge_values = model(data)

    path = get_path_from_values(data.edge_index, edge_values)
    print(f"{path} \n\n {tnu.verify_path(path)} \n\n {tnu.get_dot_from_path(path)}")

    print("DONE")