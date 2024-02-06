import numpy as np
import math
import random
import file_util as fu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Batch
#import tensor_network_util as tnu

class EdgePredictionGNN(nn.Module):
    def __init__(self, hidden_size, node_layers, edge_layer):
        super(EdgePredictionGNN, self).__init__()

        self.gate_index = {'H': 0, 'X': 1, 'CX_0': 2, 'CX_1': 3, 'RZ': 4, 'S': 5, 'CZ_0': 6, 'CZ_1': 7, 'U3': 8}

        self.dim_layer = nn.Linear(1, int(hidden_size / 2))
        self.emb_layer = nn.Embedding(len(self.gate_index), int(hidden_size / 2))
        self.node_layers = nn.ModuleList([gnn.GATConv(hidden_size, hidden_size) if i % 4 == 3 else gnn.SAGEConv(hidden_size, hidden_size)  
                                          for i in range(node_layers)])
        self.edge_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(edge_layer)])
        self.edge_layer = nn.Linear(hidden_size, 1)

    def get_edge_features(self, x, edge_index):
        src, dst = edge_index
        return x[src] + x[dst]

    def forward(self, data):
        edge_index = data.edge_index
        x = torch.tensor([[len(s) for s in data.shape]], dtype=torch.float).transpose(0,1)

        gate_indices = torch.tensor([self.gate_index[gate] for gate in data.gate], dtype=torch.long)

        x_emb = self.emb_layer(gate_indices)
        x_dim = self.dim_layer(x)
        x = torch.cat((x_dim,x_emb), dim=1)

        for layer in self.node_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        x = self.get_edge_features(x, edge_index)
        for layer in self.edge_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.edge_layer(x)

        return x


class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    model.eval()
    return model

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

def get_path(model, graph_data):
    return get_path_from_values(graph_data.edge_index, model(graph_data))

def prepare_graph(graph, target):
    data = from_networkx(graph)
    data[target] = (data[target] / max(data[target])).unsqueeze(1)
    return data

def prepare_graphs(graphs, target):
    return [prepare_graph(graph, target) for graph in graphs]

def training(data, data_loader, validation_graphs): 
    
    print("Building Model")

    model = EdgePredictionGNN(data["hidden_size"], data["node_layers"], data["edge_layers"])
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=data["lr"])

    data["loss"] = []
    data["val_loss"] = []

    print("Training")

    for epoch in range(data["num_epochs"]):
        model.train()
        running_loss = 0.0

        for batch in data_loader:        
            optimizer.zero_grad()

            loss = torch.Tensor([0.0])

            for graph in batch.to_data_list():
                outputs = model(graph)
                loss += loss_function(outputs, graph[data["target"]])

            loss.backward()
            optimizer.step()

            # Update running loss
            l = loss.item()
            running_loss += l

        model.eval()
        validation_loss = torch.Tensor([0.0])
        for validation_graph in validation_graphs:
            validation_loss += loss_function(model(validation_graph), validation_graph[data["target"]]).item() * 1000

        validation_loss = validation_loss.item() / len(validation_graphs)
        average_loss = running_loss / len(graphs) * 1000

        data["loss"].append(average_loss)
        data["val_loss"].append(validation_loss)

        print(f'Epoch [{epoch + 1}/{data["num_epochs"]}], Loss: {average_loss:.2f}, validation Loss: {validation_loss:.2f}')


if __name__ == "__main__":
    print("Loading")
    graphs = fu.load_all_nx_graphs("graphs\\betweenness") #"graphs\\random_greedy"
    #graphs = [fu.load_nx_graph("C:\\Users\\simon\\Documents\\GitHub\\P10\\graphs\\random_greedy\\graph_dj_q5.gml")]

    
    data = [{
        "experiment":"lr3",
        "num_epochs": 200,
        "batch_size": 60,
        "hidden_size": 64,
        "node_layers": 10,
        "edge_layers": 3,
        "lr":10**(-2.925),
        "target": "betweenness" #"random_greedy"
    } for lr in range(20)]

    graphs = prepare_graphs(graphs, data[0]["target"])
    

    for i, d in enumerate(data):
        d["run"] = i
        validation_graphs = [graphs.pop(random.randint(0, len(graphs) - 1))  for _ in range(int(len(graphs) * 0.1))]
        data_loader = DataLoader(GraphDataset(graphs), batch_size=data[0]["batch_size"], shuffle=True, collate_fn= lambda batch: Batch.from_data_list(batch))

        training(d, data_loader, validation_graphs)
        fu.save_to_json(f"experiment_data\\{d['experiment']}", f"experiment_{i}.json", d)
        print("Saved")
