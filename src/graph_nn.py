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

class EdgePredictionGNN(nn.Module):
    def __init__(self, hidden_size, node_layers, edge_layer):
        super(EdgePredictionGNN, self).__init__()

        self.gate_index = {'H': 0, 'CX_0': 1, 'CX_1': 2, 'RZ': 3, 'RX': 4, 'U3': 5, 'RY': 6, 'S': 7, 'X': 8, 
                           'CZ_0': 9, 'CZ_1': 10, 'CY_0': 11, 'CY_1': 12, 'Y': 13, 'Z': 14, 'T': 15}
        

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
    
class IterativeGNN(nn.Module):
    def __init__(self, hidden_size, node_layers, edge_layer):
        super(IterativeGNN, self).__init__()

        self.gate_index = {'H': 0, 'CX_0': 1, 'CX_1': 2, 'RZ': 3, 'RX': 4, 'U3': 5, 'RY': 6, 'S': 7, 'X': 8, 
                           'CZ_0': 9, 'CZ_1': 10, 'CY_0': 11, 'CY_1': 12, 'Y': 13, 'Z': 14, 'T': 15}
        

        self.dim_layer = nn.Linear(1, int(hidden_size / 2))
        self.emb_layer = nn.EmbeddingBag(len(self.gate_index), int(hidden_size / 2), mode="mean")
        self.node_layers = nn.ModuleList([gnn.GATConv(hidden_size, hidden_size) if i % 4 == 3 else gnn.SAGEConv(hidden_size, hidden_size)  
                                          for i in range(node_layers)])
        self.edge_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(edge_layer)])
        self.edge_layer = nn.Linear(hidden_size, 1)

    def get_edge_features(self, x, edge_index):
        src, dst = edge_index
        return x[src] + x[dst]

    def create_bag_embeddings(_, gate_index_bags):
        # Convert the list of lists to a list of tensors
        list_of_tensors = torch.cat([torch.tensor(lst, requires_grad=False) for lst in gate_index_bags])

        # Create offsets tensor based on the lengths of the original sequences
        offsets = torch.cumsum(torch.tensor([0] + [len(lst) for lst in gate_index_bags[:-1]]), dim=0)

        return list_of_tensors, offsets

    def forward(self, data, target = None, loss_function = None, learn = True):
        data = data.clone()
        shapes = torch.tensor([[len(s) for s in data.shape]], dtype=torch.float).transpose(0,1)
        edges = data[target] if target is not None else None

        path = []
        loss = 0
        gate_index_bags = [[self.gate_index[gate]] for gate in data.gate]

        for i in range(data.num_nodes - 1):
            if i % 100 == 0:
                print(f"Iteration: {i}")
            gate_indices, offsets = self.create_bag_embeddings(gate_index_bags)

            x_emb = self.emb_layer(gate_indices, offsets)
            x_dim = self.dim_layer(shapes)
            x = torch.cat((x_dim,x_emb), dim=1)

            for layer in self.node_layers:
                x = layer(x, data.edge_index)
                x = F.relu(x)
            x = self.get_edge_features(x, data.edge_index)
            for layer in self.edge_layers:
                x = layer(x)
                x = F.relu(x)
            x = self.edge_layer(x)

            
            edge = torch.argmin(x) 
            
            if target is not None:
                r = torch.ones_like(edges, requires_grad=False, dtype=torch.float)
                r[edges != i] = 0

                x = x.squeeze()
                if x.shape[0] != r.shape[0]:
                    print(f"DELETE: {data.name} <---------------------")
                    return None, torch.tensor([0.0])
                current_loss = loss_function(x, r)
                if learn:
                    current_loss.backward(retain_graph = False)
                loss += current_loss.item()

                edge = torch.argmin(edges)
                edges = edges[edges != i]

                

            n0 = data.edge_index[0][edge].item()
            n1 = data.edge_index[1][edge].item()
            path.append((n1, n0))

            shapes = self.contract(shapes, gate_index_bags, data, n0, n1)

        return path, loss / (data.num_nodes - 1)
    
    def contract(self, shapes, gate_index_bags, data, n0, n1):
        data.edge_index = data.edge_index.clone()
        shapes = shapes.clone()

        j = 0
        sources = []
        targets = []

        with torch.no_grad():
            for i in range(data.num_edges):
                if(data.edge_index[0][i] == n1):
                    if(data.edge_index[1][i] != n0):
                        data.edge_index[0][i] = n0
                        shapes[n0] += 1
                        ...
                    else:
                        sources.append(data.edge_index[0][j:i])
                        targets.append(data.edge_index[1][j:i])
                        j = i+1
                        data.num_edges -= 1
                if(data.edge_index[1][i] == n1):
                    if(data.edge_index[0][i] != n0):
                        data.edge_index[1][i] = n0
                        ...
                    else:
                        sources.append(data.edge_index[0][j:i])
                        targets.append(data.edge_index[1][j:i])
                        j = i+1
                        data.num_edges -= 1

            sources.append(data.edge_index[0][j:])
            targets.append(data.edge_index[1][j:])

            gate_index_bags[n0] = gate_index_bags[n0] + gate_index_bags[n1]
            data.edge_index = torch.stack([torch.cat(sources), torch.cat(targets)])

            return shapes

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
    #edges = data[target]
    #data[target] = []
    
    #print(f"{data.num_nodes}")
    #for i in range(data.num_nodes - 1):
    #    r = torch.ones_like(edges)
    #    r[edges != i] = 0
    #    data[target].append(r)
    #    edges = edges[edges != i]
    return data

def prepare_graphs(graphs, target):
    return [prepare_graph(graph, target) for graph in graphs]

def training(data, data_loader, validation_graphs, model = None, iterative = True): 
    
    print("Building Model")

    if(model is None):
        model = IterativeGNN(data["hidden_size"], data["node_layers"], data["edge_layers"]) if iterative else EdgePredictionGNN(data["hidden_size"], data["node_layers"], data["edge_layers"])
    
    loss_function = nn.CrossEntropyLoss() if iterative else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=data["lr"])
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    data["loss"] = []
    data["val_loss"] = []

    min_val_loss = float("inf")
    epochs_since_improvement = 0

    print("Training")

    for epoch in range(data["num_epochs"]):
        model.train()
        running_loss = 0.0

        for batch in data_loader:        
            optimizer.zero_grad()

            loss = torch.Tensor([0.0])

            for graph in batch.to_data_list():
                if iterative:
                    print(f"Graph Nodes: {graph.num_nodes}")
                    _, l = model(graph, data["target"], loss_function, optimizer)
                    loss += l
                else:
                    outputs = model(graph)
                    loss += loss_function(outputs, graph[data["target"]])

            if not iterative: 
                loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        model.eval()
        validation_loss = torch.Tensor([0.0])
        if iterative:
            for validation_graph in validation_graphs:
                print(f"Val Graph Nodes: {graph.num_nodes}")
                _, l = model(graph, data["target"], loss_function, None)
                loss += l
        
        else:
            for validation_graph in validation_graphs:
                validation_loss += loss_function(model(validation_graph), validation_graph[data["target"]]).item() * 1000

        validation_loss = validation_loss.item() / len(validation_graphs)
        average_loss = running_loss / len(data_loader.dataset) * 1000

        data["loss"].append(average_loss)
        data["val_loss"].append(validation_loss)

        if(validation_loss < min_val_loss):
            min_val_loss = validation_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

            if(epochs_since_improvement >= 10):
                print("Early Stopping...")
                break

        print(f'Epoch [{epoch + 1}/{data["num_epochs"]}], Loss: {average_loss:.2f}, validation Loss: {validation_loss:.2f}')

    return model


def run():

    print("Loading")
    training_graphs = fu.load_all_nx_graphs("dataset/mini_split/train") #"graphs\\random_greedy"
    validation_graphs =  fu.load_all_nx_graphs("dataset/mini_split/val")
    #graphs = [fu.load_nx_graph("C:\\Users\\simon\\Documents\\GitHub\\P10\\graphs\\random_greedy\\graph_dj_q5.gml")]


    
    data = [{
        #"load_experiment":"bbds2",
        #"load_name": "experiment_n2",
        "iterative": True,
        "experiment":"mini_iteration_0",
        "save_model":False,
        "num_epochs": 10,
        "batch_size": 60,
        "hidden_size": 64,
        "node_layers": 10,
        "edge_layers": 3,
        "lr":10**(-2.5),
        "target": "betweenness", #"random_greedy"
        "run": i,
        "run_name":  f"experiment_{i}"
    } for i in range(1)]

    print("Preparing Graphs")
    training_graphs = prepare_graphs(training_graphs, data[0]["target"])
    validation_graphs = prepare_graphs(validation_graphs, data[0]["target"])

    for i, d in enumerate(data):
        print("Splitting Data")
        #graphs = [g for g in all_graphs]
        #validation_graphs = [graphs.pop(random.randint(0, len(graphs) - 1))  for _ in range(int(len(graphs) * 0.1))]
        data_loader = DataLoader(GraphDataset(training_graphs), batch_size=data[0]["batch_size"], shuffle=True, collate_fn= lambda batch: Batch.from_data_list(batch))

        if "load_experiment" in d:
            print("Loading Model")
            model = torch.load(fu.get_path("experiment_data/" + d["load_experiment"] + "/models/" + d["load_name"]))
        else:
            model = None

        model = training(d, data_loader, validation_graphs, model = model, iterative= "iterative" in d and d["iterative"])
        fu.save_to_json(f"experiment_data/{d['experiment']}", d["run_name"] + ".json", d)
        if d["save_model"]:
            fu.save_model(model, f"experiment_data/{d['experiment']}/models", d["run_name"] + ".pt")
        print("Saved")

if __name__ == "__main__":
    run()
    
    #graphs = fu.load_all_nx_graphs("dataset/betweenness")
    #graphs = prepare_graphs(graphs, "betweenness")
    #model = IterativeGNN(32, 10, 3)
    #model.eval()
    #graphs = sorted(graphs, key = lambda g: g.num_nodes)
    #path, loss = model(graphs[0])
    #print("Done")

