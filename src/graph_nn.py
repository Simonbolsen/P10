import numpy as np
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
        self.node_layers = nn.ModuleList([gnn.SAGEConv(hidden_size, hidden_size) for _ in range(node_layers)])
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

if __name__ == "__main__":
    print("Loading")
    graphs = fu.load_all_nx_graphs("graphs\\random_greedy") #"graphs\\random_greedy"
    #graphs = [fu.load_nx_graph("C:\\Users\\simon\\Documents\\GitHub\\P10\\graphs\\random_greedy\\graph_dj_q5.gml")]
    
    print("Building Model")

    num_epochs = 10000
    batch_size = 10
    hidden_size = 128
    node_layers = 20
    edge_layers = 5
    target = "random_greedy"

    graphs = prepare_graphs(graphs, target)
    verification_graph = graphs.pop(7)  
    data_loader = DataLoader(GraphDataset(graphs), batch_size=batch_size, shuffle=True, collate_fn= lambda batch: Batch.from_data_list(batch))
    model = EdgePredictionGNN(hidden_size, node_layers, edge_layers)
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        losses = []

        for batch in data_loader:        
            optimizer.zero_grad()

            loss = torch.Tensor([0.0])

            for graph in batch.to_data_list():
                outputs = model(graph)
                loss += loss_function(outputs, graph[target])

            loss.backward()
            optimizer.step()

            # Update running loss
            l = loss.item()
            running_loss += l
            losses.append(l)

        model.eval()
        verification_loss = loss_function(model(verification_graph), verification_graph[target].float().unsqueeze(1)).item() * 1000

        # Print the average loss for the epoch
        average_loss = running_loss / len(graphs) * 1000
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.2f}, Verification Loss: {verification_loss:.2f}')
        if epoch % 100 == 99:
            print(losses)

    #print(f"{path} \n\n {tnu.verify_path(path)} \n\n {tnu.get_dot_from_path(path)}")
    