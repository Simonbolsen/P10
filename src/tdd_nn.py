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
from random import random as rand

GATE_INDICES = {'H': 0, 'CX': 1, 'CNOT':1, 'RZ': 2, 'RX': 3, 'U3': 4, 'RY': 5, 'S': 6, 'X': 7, 
                        'CZ': 8, 'CY': 9, 'Y': 10, 'Z': 11, 'T': 12}
GATE_SIZES = {'CX': 6, 'CNOT': 6, 'CZ': 6, 'RZ': 4, 'S': 4, 'H': 3, 'Y': 4, 'Z': 4, 'X': 4, 'CY': 6, 'T': 4, 'RY': 3, 'RX': 4, "U3": 4}

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA version:", torch.version.cuda)
    print("Number of CUDA devices:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device properties:", torch.cuda.get_device_properties(device))
else: 
    device = torch.device("cpu")
    print("Using CPU")

def get_elapsed_time(start_time, end_time):
    t = end_time - start_time
    return f"{int(t.seconds / 60)}m {((t.seconds + t.microseconds / 1000000) % 60):.3f}s"

def get_tensors(tensor_network):
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
    
    return tensors, index_sets

class TDDPredicter(nn.Module):
    def __init__(self, hidden_size, depth, dropout_probability):
        super(TDDPredicter, self).__init__()        

        self.hidden_size = hidden_size
        self.depth = depth

        self.input_layer = nn.Linear(2 + 14, hidden_size)
        #self.input_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
        self.shared_index_layer = nn.Linear(1, hidden_size)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
        self.dropout_layer = nn.Dropout(p = dropout_probability)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, l, r, s):
        x = self.input_layer(l) + self.input_layer(r) + self.shared_index_layer(s)

        for _, layer in enumerate(self.linear_layers):
            y = x
            x = self.dropout_layer(x)
            x = layer(x)
            x = self.relu(x)
            x = x + y

        x = self.output_layer(x)

        return x
    
def call_model(model, input):
    return model(input["left_values"], input["right_values"], input["shared_values"])

def train_model(data, training_data, validation_data, model = None):
    print("Building Model")

    if(model is None):
        if "model" not in data or data["model"] == "predicter":
            model = TDDPredicter(data["hidden_size"], data["depth"], data["dropout_probability"]).to(device)
        elif data["model"] == "baseline":
            ...#model = TDDBaseline(data["hidden_size"], data["depth"])
        else:
            print(f"Invalid Model {data['model']}")

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, amsgrad="amsgrad" in data and data["amsgrad"], 
                                 weight_decay=data["weight_decay"] if "weight_decay" in data else 0)

    data["loss"] = []
    data["val_loss"] = []

    min_val_loss = float("inf")
    epochs_since_improvement = 0
    best_model_state = None

    print("Training")

    # Move input data to GPU before the loop

    for epoch in range(data["num_epochs"]):
        start_time = dt.today()
        model.train()
        running_loss = 0.0

        learning_rate = 10**(data["lr"] + data["lr_decay"] / (1 + data["lr_decay_speed"] * epoch ** 2))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        

        for batch in training_data:        

            optimizer.zero_grad()

            outputs = call_model(model, batch)
            loss = loss_function(outputs, batch["target"])

            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

        model.eval()
        validation_loss = 0
        batch_data = None
        for val_batch in validation_data:
            
            output = call_model(model, val_batch)
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

        
        print(f'Epoch [{epoch + 1}/{data["num_epochs"]}], \tLoss: {int(average_loss)}, \tValidation Loss: {int(validation_loss)},'+ 
              f'\tLearning rate: {math.log10(learning_rate):.2f}, \tElapsed Time: {get_elapsed_time(start_time, dt.today())}  {"<-" if epochs_since_improvement == 0 else ""}')
 
    return model, best_model_state

def load_model(path):
    saved_dict = torch.load(path)
    model = TDDPredicter(64, int(len([0 for i in saved_dict.keys() if "linear_layers" in i])/2), 0.008)
    model.load_state_dict(saved_dict)
    model.eval()
    return model

def torchify(data):
    torched = []

    for d in data:
        torched.append(
            {"left_values":torch.tensor(d["left_values"]), 
                "right_values":torch.tensor(d["right_values"]),
                "shared_values":torch.tensor(d["shared_values"]),
                "target":torch.tensor(d["target"]),})
    return torched

def cuda_collate(data):
    moved_batch = {"left_values": [], "right_values": [], "shared_values": [], "shared_values": [], "target": []}
    for element in data:
        for key in moved_batch:
            moved_batch[key].append(element[key])

    for key in moved_batch:
        moved_batch[key] = torch.stack(moved_batch[key]).to(device)
    return moved_batch

def get_dataloaders(dataset, batch_size):
    training_data = fu.load_single_json(fu.get_path(f"dataset/{dataset}/train.json"))
    validation_data =  fu.load_single_json(fu.get_path(f"dataset/{dataset}/val.json"))

    data_loader = DataLoader(torchify(training_data), batch_size=batch_size, shuffle=True, collate_fn=cuda_collate)
    val_data_loader = DataLoader(torchify(validation_data), batch_size=len(validation_data), shuffle=False, collate_fn=cuda_collate)

    return data_loader, val_data_loader

def run():
    settings = [{
        #"load_experiment":"bbds2",
        #"load_name": "experiment_n2",
        "experiment":"tdd_mk_V_c_model",
        "save_model":True,
        "model":"predicter",#"baseline",
        "dropout_probability": 0.001 * 2**(2 / 3),
        "num_epochs": 1000,
        "batch_size": int(2**(11)),
        "hidden_size": int(2**(6)),
        "depth": 10,
        "lr":-(2.9), # 
        "lr_decay": 0.8,
        "lr_decay_speed": 0.04,
        "weight_decay":0,
        "early_stopping":20,
        "warmup":10,
        "run": i,
        "run_name":  f"model_{i}"
    } for i in range(10)]

    print("Loading")
    dataset = "TSP5"

    data_loader, val_data_loader = get_dataloaders(dataset, settings[0]["batch_size"])

    for s in settings:

        begin_time = dt.today()
        s["begin_time"] = begin_time.isoformat()

        if "load_experiment" in s:
            print("Loading Model")
            model = torch.load(fu.get_path("experiment_data/" + s["load_experiment"] + "/models/" + s["load_name"]))
        else:
            model = None

        print(f"CUDA memory usage. Current: {torch.cuda.memory_allocated()}, Max: {torch.cuda.max_memory_allocated()}, Total: {torch.cuda.memory_reserved()}")
        model, state = train_model(s, data_loader, val_data_loader, model = model)

        end_time = dt.today()
        s["end_time"] = end_time.isoformat()
        
        fu.save_to_json(f"experiment_data/{s['experiment']}", s["run_name"], s)
        if s["save_model"]:
            model.load_state_dict(state)
            model.eval()
            path = f"experiment_data/{s['experiment']}/models"
            fu.save_model(model, path, s["run_name"])
            fu.save_jit_model(model, path, s["run_name"] + "_jit")
        print(f"Saved at {s['end_time'].replace('T', ' ')}, elapsed time: {get_elapsed_time(begin_time, end_time)}")
        

if __name__ == "__main__":
    #fu.process_all_data(GATE_INDICES, 20, 5)
    run()
    ...