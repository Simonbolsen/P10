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
GATE_SIZES = {'CX': 6, 'CZ': 6, 'RZ': 4, 'S': 4, 'H': 3, 'Y': 4, 'Z': 4, 'X': 4, 'CY': 6, 'T': 4, 'RY': 3, 'RX': 4, "U3": 4}

class TDDPredicter(nn.Module):
    def __init__(self, hidden_size, depth, dropout_probability):
        super(TDDPredicter, self).__init__()        

        self.hidden_size = hidden_size
        self.depth = depth

        self.batch_norm = nn.BatchNorm1d(2 + len(GATE_INDICES))
        self.batch_norm_shared = nn.BatchNorm1d(1)
        self.input_layer = nn.Linear(2 + len(GATE_INDICES), hidden_size)
        #self.input_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])
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
            y = x
            x = self.dropout_layer(x)
            x = self.linear_layers[i](x)
            x = self.relu(x)
            x = x + y

        x = self.output_layer(x)

        return x
    
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, amsgrad="amsgrad" in data and data["amsgrad"], 
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

        learning_rate = 10**(data["lr"] + data["lr_decay"] / (1 + data["lr_decay_speed"] * epoch ** 2))
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        

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

        print(f'Epoch [{epoch + 1}/{data["num_epochs"]}], \tLoss: {average_loss:.2f}, \tValidation Loss: {validation_loss:.2f}, \tLearning rate: {math.log10(learning_rate):.2f}')

    return best_model_state

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



def run():
    settings = [{
        #"load_experiment":"bbds2",
        #"load_name": "experiment_n2",
        "experiment":"tdd_lr_decay_1",
        "save_model":False,
        "model":"predicter",#"baseline",
        "dropout_probability": 0.001 * 2**(2 / 3),
        "num_epochs": 1000,
        "batch_size": 90,
        "hidden_size": 90,
        "depth": 14,
        "lr":-(3.6),
        "lr_decay": 0.5 + 0.2 * j,
        "lr_decay_speed": 0.01 * i,
        "weight_decay":0,
        "early_stopping":20,
        "warmup":20,
        "run": 0,
        "run_name":  f"model_{i}_{j}"
    } for i in range(6) for j in range(6)]

    print("Loading")
    dataset = "TSP3"
    training_data = fu.load_single_json(fu.get_path(f"dataset/{dataset}/train.json"))
    validation_data =  fu.load_single_json(fu.get_path(f"dataset/{dataset}/val.json"))

    data_loader = DataLoader(torchify(training_data), batch_size=settings[0]["batch_size"], shuffle=True)
    val_data_loader = DataLoader(torchify(validation_data), batch_size=len(validation_data), shuffle=False)

    for s in settings:

        s["begin_time"] = dt.today().isoformat()

        if "load_experiment" in s:
            print("Loading Model")
            model = torch.load(fu.get_path("experiment_data/" + s["load_experiment"] + "/models/" + s["load_name"]))
        else:
            model = None

        model = train_model(s, data_loader, val_data_loader, model = model)

        s["end_time"] = dt.today().isoformat()
        
        fu.save_to_json(f"experiment_data/{s['experiment']}", s["run_name"], s)
        if s["save_model"]:
            fu.save_model(model, f"experiment_data/{s['experiment']}/models", s["run_name"])
        print(f"Saved: {s['end_time']}")
        

if __name__ == "__main__":
    run()
    ...