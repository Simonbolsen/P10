import json
import os
import networkx as nx
from tqdm import tqdm
import torch
import random
import math

def split(folder):
    path = os.scandir(get_path(folder))
    for sf in tqdm(path):
        data = load_json(sf.path)
        file = {"data": data, "name": sf.name[:-5]}
        
        if(random.random() < 0.1):
            save_to_json("dataset/TSP2/val", file["name"], file)
        else:
            save_to_json("dataset/TSP2/train", file["name"], file)
        
def remove_duplicates(dataset):
    seen_data = set()
    non_duplicate_data = []

    for item in tqdm(dataset):
        # Assuming each item is a tensor
        item_tuple = tuple(item["left_values"] + item["right_values"] + item["shared_values"] + item["target"])
        inv_tuple  = tuple(item["right_values"] + item["left_values"] + item["shared_values"] + item["target"])

        # Check if the item is already seen
        if item_tuple not in seen_data and inv_tuple not in seen_data:
            seen_data.add(item_tuple)
            non_duplicate_data.append(item)

    return non_duplicate_data

def prepare(d, gate_indices):
    l = d["left"]
    r = d["right"]
    o = d["result"]

    lg = [0 for _ in gate_indices]
    rg = [0 for _ in gate_indices]

    if "time" in d:
        l_gates = [g["name"] for g in l["gates"]]
        r_gates = [g["name"] for g in r["gates"]]
    else:
        l_gates = l["gates"]
        r_gates = r["gates"]

    for gate in l_gates:
        lg[gate_indices[gate.upper()]] += 1

    for gate in r_gates:
        rg[gate_indices[gate.upper()]] += 1

    return {"left_values":[math.log2(l["nodes"]), len(l["indices"])] + lg, 
                "right_values":[math.log2(r["nodes"]), len(r["indices"])] + rg, 
                "shared_values":[(len(l["indices"]) + len(r["indices"]) - len(o["indices"])) / 2],
                "target":[math.log2(o["nodes"])]}

def prepare_all_data(data, gate_indices):
    p = []
    for d2 in data:
        for d1 in d2["data"]:
            if type(d1) is list:
                for d in d1:
                    p.append(prepare(d, gate_indices))
            else:
                p.append(prepare(d1, gate_indices))
    return p

def process_data(path):
    data = load_all_json(path)
    data = prepare_all_data(data)
    return remove_duplicates(data)

def process_all_data():
    paths = ["dataset/TSP2/b1", "dataset/TSP2/b2", "dataset/TSP2/b3", "dataset/TSP2/b4", "dataset/TSP2/b5", "dataset/TSP2/b6", 
             "dataset/TSP2/b7", "dataset/TSP2/b8", "dataset/TSP2/train", "dataset/TSP2/val"]
    final_data = []

    for p in paths:
        final_data.extend(process_data(p))
        
    final_data = remove_duplicates(final_data)
    print(len(final_data))

    train = []
    val = []

    for f in final_data:
        if random.random() < 0.1:
            val.append(f)
        else:
            train.append(f)

    print(f"Train: {len(train)}, Val: {len(val)}")

    save_to_json(f"dataset/TSP3", "train", train)
    save_to_json(f"dataset/TSP3", "val", val)

def get_path(folder) :
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', folder))

def save_to_json(folder, file_name, object):
    folder_path = get_path(folder)

    make_folder(folder_path)
    
    with open(os.path.join(folder_path, file_name + ".json"), 'w+') as outfile:
        json.dump(json.dumps(object), outfile)

def load_json(file_path):
    os_path = os.path.join(os.path.realpath(__file__), '..', file_path)
    with open(os_path, 'r') as json_file:
        data = json_file.read()
    return json.loads(data)

def load_all_json(folder):
    return load_rec(get_path(folder), is_file_type(".json"), load_single_json)  

def load_all_file_paths(folder):
    return load_rec(get_path(folder), is_file_type(".qasm"), lambda x:x)   

def load_all_nx_graphs(folder):
    return load_rec(get_path(folder), is_file_type(".gml"), lambda x:load_nx_graph(x.path))  

def load_rec(file, func, load):
    files = []
    path = os.scandir(file)
    for sf in tqdm(path):
        if sf.is_dir():
            files += load_rec(sf, func, load)
        elif func(sf):
            files.append(load(sf))

    return files

def is_file_type(ftype):
    return lambda file: os.path.splitext(file)[1] == ftype

def load_single_json(file):
    with open(file, 'r') as json_file:
        j = json.loads(json_file.read())
        return json.loads(j) if type(j) is str else j

def save_nx_graph(graph: nx.Graph, path: str, file_name: str):
    final_path = file_name + ".gml" if path == "" else os.path.join(path, file_name + ".gml")
    nx.write_gml(graph, final_path)

def load_nx_graph(path: str):
    graph = nx.read_gml(path)
    return graph

def save_model(model, folder, file_name):
    path = get_path(folder)
    make_folder(path)

    torch.save(model, os.path.join(path, file_name + ".pt"))

def load_model(path):
    torch.load(path)

def make_folder(folder_path):
    if not os.path.exists(folder_path):
        print("==> folder to save embedding does not exist... creating folder...")
        print("   ==> folder path: ", folder_path)
        os.mkdir(folder_path)