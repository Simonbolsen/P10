import numpy as np
import math
import file_util as fu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime as dt
import torch.nn.init as init
import copy
import tensor_network_util as tnu
from tqdm import tqdm
import time

GATE_INDICES = {'H': 0, 'CX': 1, 'RZ': 2, 'RX': 3, 'U3': 4, 'RY': 5, 'S': 6, 'X': 7, 
                        'CZ': 8, 'CY': 9, 'Y': 10, 'Z': 11, 'T': 12}
GATE_SIZES = {'CX': 6, 'CZ': 6, 'RZ': 4, 'S': 4, 'H': 3, 'Y': 4, 'Z': 4, 'X': 4, 'CY': 6, 'T': 4, 'RY': 3, 'RX': 4}

class TDDPredicter(nn.Module):
    def __init__(self, hidden_size, depth, dropout_probability):
        super(TDDPredicter, self).__init__()        

        self.hidden_size = hidden_size
        self.depth = depth

        self.batch_norm = nn.BatchNorm1d(2 + len(GATE_INDICES))
        self.batch_norm_shared = nn.BatchNorm1d(1)
        self.input_layer = nn.Linear(2 + len(GATE_INDICES), hidden_size)
        self.shared_index_layer = nn.Linear(1, hidden_size)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
        self.dropout_layer = nn.Dropout(p = dropout_probability)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        for l in self.linear_layers:
            init.kaiming_uniform_(l.weight, nonlinearity="relu")

    def forward(self, input):
        
        l = input["left_values"]
        #l = self.batch_norm()
        l = self.input_layer(l)

        r = input["right_values"]
        #r = self.batch_norm()
        r = self.input_layer(r)

        s = input["shared_values"]
        #s = self.batch_norm_shared()
        s = self.shared_index_layer(s)

        x = l + r + s

        for i in range(self.depth):
            x = self.dropout_layer(x)
            x = self.linear_layers[i](x)
            x = self.relu(x)

        x = self.output_layer(x)

        return x
    
class TDDBaseline(nn.Module):
    def __init__(self, hidden_size, depth):
        super(TDDBaseline, self).__init__()

        self.input_layer = nn.Linear(3, hidden_size)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, input):

        if len(input["left_values"].shape) == 1:
            x = torch.tensor([input["left_values"][0], input["right_values"][0], input["shared_values"][0]])
        else:
            x =  torch.stack([input["left_values"][:,0], input["right_values"][:,0], input["shared_values"][:,0]], dim = 1)

        x = self.input_layer(x)

        for layer in self.linear_layers:
            x = layer(x)
            x = self.relu(x)

        x = self.output_layer(x)

        return x
    
class TddDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.output_data[idx]

def train_model(data, training_data, validation_data, model = None):
    print("Building Model")

    if(model is None):
        if "model" not in data or data["model"] == "predicter":
            model = TDDPredicter(data["hidden_size"], data["depth"], data["dropout_probability"])
        elif data["model"] == "baseline":
            model = TDDBaseline(data["hidden_size"], data["depth"])
        else:
            print(f"Invalid Model {data['model']}")

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=data["lr"], amsgrad="amsgrad" in data and data["amsgrad"], 
                                 weight_decay=data["weight_decay"] if "weight_decay" in data else 0)

    data["loss"] = []
    data["val_loss"] = []

    min_val_loss = float("inf")
    epochs_since_improvement = 0
    best_model_state = None

    print("Training")

    for epoch in range(data["num_epochs"]):
        model.train()
        running_loss = 0.0

        for batch in training_data:        
            optimizer.zero_grad()

            outputs = model(batch)
            loss = loss_function(outputs, batch["target"])
                
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        model.eval()
        validation_loss = 0
        batch_data = None
        for val_batch in validation_data:
            output = model(val_batch)
            val_loss = loss_function(output, val_batch["target"]).item()
            validation_loss += val_loss
            batch_data = (output, val_batch)

        validation_loss = validation_loss / len(validation_data) * 10000
        average_loss = running_loss / len(training_data) * 10000

        data["loss"].append(average_loss)
        data["val_loss"].append(validation_loss)

        if(validation_loss < min_val_loss):
            min_val_loss = validation_loss
            epochs_since_improvement = 0
            data["batch_data"] = (batch_data[0].tolist(), batch_data[1]["left_values"].tolist(), 
                                  batch_data[1]["right_values"].tolist(), batch_data[1]["shared_values"].tolist(),
                                  batch_data[1]["target"].tolist())
            if "save_model" in data and data["save_model"]:
                best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_since_improvement += 1

            if(epochs_since_improvement >= data["early_stopping"] and epoch >= data["warmup"]):
                print("Early Stopping...")
                break

        print(f'Epoch [{epoch + 1}/{data["num_epochs"]}], Loss: {average_loss:.2f}, validation Loss: {validation_loss:.2f}')

    return best_model_state

def prepare(d):
    l = d["left"]
    r = d["right"]
    o = d["result"]

    lg = [0 for _ in GATE_INDICES]

    for gate in l["gates"]:
        lg[GATE_INDICES[gate]] += 1

    rg = [0 for _ in GATE_INDICES]

    for gate in l["gates"]:
        rg[GATE_INDICES[gate]] += 1

    return {"left_values":torch.tensor([math.log2(l["nodes"]), len(l["indices"])] + lg), 
                "right_values":torch.tensor([math.log2(r["nodes"]), len(r["indices"])] + rg), 
                "shared_values":torch.tensor([(len(l["indices"]) + len(r["indices"]) - len(o["indices"])) / 2]),
                "target":torch.tensor([math.log2(o["nodes"])])}

def prepare_all_data(data):
    p = []
    for d2 in data:
        for d1 in d2["data"]:
            if type(d1) is list:
                for d in d1:
                    p.append(prepare(d))
            else:
                p.append(prepare(d1))
    return p


def load_model(path):
    saved_dict = torch.load(path)
    model = TDDPredicter(64, 4, 0.008)
    model.load_state_dict(saved_dict)
    model.eval()
    return model

def get_path(model, tensor_network, print_sizes = False):
    path = []

    edges = {}
    tensors = {}
    index_sets = {}

    for tid, tensor in tensor_network.tensor_map.items():
        g = [0 for _ in GATE_INDICES]

        size = 0

        for t in tensor.tags:
            if t in GATE_INDICES:
                g[GATE_INDICES[t]] += 1
                size += GATE_SIZES[t]

        size = math.log2(size) if size > 0 else len(tensor.inds)
        tensors[tid] = torch.tensor([size, len(tensor.inds)] + g, dtype=torch.float)

        index_sets[tid] = set()

    for o in tensor_network.ind_map.values():
        key = (min(o), max(o))
        if key[0] != key[1]:
            if key in edges:
                edges[key] += 1
            else:
                edges[key] = 1
            
            index_sets[key[0]].add(key[1])
            index_sets[key[1]].add(key[0])

    progress_bar = tqdm(total=len(tensors) - 1, desc="Countdown", unit="step")

    prediction_times = 0
    cleanup_time = 0

    new_edges = list(edges.keys())
    edge_predictions = {}

    while len(edges) > 0:
        progress_bar.update(1)

        prediction_times -= time.time()

        for e in new_edges:
            input = {"left_values":tensors[e[0]], "right_values": tensors[e[1]], "shared_values": torch.tensor([i], dtype=torch.float)}
            edge_predictions[e] = model(input).item()

        prediction_times += time.time()

        step, prediction = min(edge_predictions.items(), key=lambda x:x[1])
        path.append(step)

        if print_sizes:
            print(prediction)

        cleanup_time -= time.time()

        i = edges.pop(step)
        edge_predictions.pop(step)

        tensor = tensors.pop(step[0])
        tensor = tensors[step[1]] + tensor
        tensor[0] = prediction
        tensor[1] -= i * 2
        tensors[step[1]] = tensor

        index_sets[step[0]].remove(step[1])
        index_sets[step[1]].remove(step[0])

        for i in index_sets[step[0]]:
            key = (min(step[0], i), max(step[0], i))

            if key[0] != key[1] and step[1] != i:
                index_sets[i].add(step[1])
                index_sets[i].remove(step[0])
                index_sets[step[1]].add(i)

                n = edges.pop(key)
                edge_predictions.pop(key)

                k = key[1] if key[0] == step[0] else key[0]
                new_key = (min(k, step[1]), max(k, step[1]))

                if new_key in edges:
                    edges[new_key] += n
                else:
                    edges[new_key] = n
        cleanup_time += time.time()

    print(f"Prediction Time: {prediction_times}, Cleanup Time: {cleanup_time}")

    return path

def run():

    print("Loading")
    dataset = "TSP2"
    training_data = fu.load_all_json(f"dataset/{dataset}/train")
    validation_data =  fu.load_all_json(f"dataset/{dataset}/val")

    data = [{
        #"load_experiment":"bbds2",
        #"load_name": "experiment_n2",
        "experiment":"tdd_mk2",
        "save_model":True,
        "model":"predicter",#"baseline",
        "dropout_probability": 0.008,
        "num_epochs": 1000,
        "batch_size": 90,
        "hidden_size": 64,
        "depth": 4,
        "lr":10**(-(2.8 + 0.05 * i)),
        "weight_decay":0,
        "early_stopping":20,
        "warmup":20,
        "run": 0,
        "run_name":  f"model_{i}"
    } for i in range(1, 10)]

    print("Preparing Data")
    training_data = prepare_all_data(training_data)
    validation_data = prepare_all_data(validation_data)

    data_loader = DataLoader(training_data, batch_size=data[0]["batch_size"], shuffle=True)
    val_data_loader = DataLoader(validation_data, batch_size=len(validation_data), shuffle=False)

    for d in data:

        d["begin_time"] = dt.today().isoformat()

        if "load_experiment" in d:
            print("Loading Model")
            model = torch.load(fu.get_path("experiment_data/" + d["load_experiment"] + "/models/" + d["load_name"]))
        else:
            model = None

        model = train_model(d, data_loader, val_data_loader, model = model)

        d["end_time"] = dt.today().isoformat()
        
        fu.save_to_json(f"experiment_data/{d['experiment']}", d["run_name"], d)
        if d["save_model"]:
            fu.save_model(model, f"experiment_data/{d['experiment']}/models", d["run_name"])
        print(f"Saved: {d['end_time']}")
        

if __name__ == "__main__":
    #fu.split("dataset/TSP2/all")
    #run()

    model = load_model(fu.get_path("experiment_data/tdd_mk2/models/model_8.pt"))
    network = tnu.get_tensor_network(tnu.get_circuit(5), False)

    path = get_path(model, network)
    print(tnu.verify_path(path))
    #print(tnu.get_dot_from_path(path))