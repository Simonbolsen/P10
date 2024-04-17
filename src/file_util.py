import json
import os
import networkx as nx
from tqdm import tqdm
import torch
import random
import math
import shutil

def move_files(source_folder, folder_min, num_files):
    # Get a list of files in the source folder
    files = os.listdir(source_folder)

    n = 0
    p = folder_min
    while n < len(files):
        p += 1
        destination_folder = f"dataset/TSP/b{p}"
        # Move num_files files from the source folder to the destination folder

        make_folder(destination_folder)

        for file_name in files[n:min(n+num_files, len(files))]:
            source_file = os.path.join(source_folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            shutil.move(source_file, destination_file)
            print(f"Moved {file_name} to {destination_folder}")
        n += num_files
    if n >= len(files):
        print(f"{destination_file} has not been filled")

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

def process_data(path, gate_indices):
    data = load_all_json(path)
    data = prepare_all_data(data, gate_indices)
    return remove_duplicates(data), len(data)

def process_all_data(gate_indices, folder_num, mk):
    final_data = []

    #sum = 0
#
    #for p in range(21,folder_num + 1):
    #    print(f"Processing: b{p}")
    #    data, original_num = process_data(f"dataset/TSP/b{p}", gate_indices)
    #    final_data.extend(data)
    #    sum += original_num

    files = load_all_json(f"dataset/TSP6")
    for f in files:
        final_data.extend(f)
        
    final_data = remove_duplicates(final_data)
    print(f"Original data: {sum}, Final data: {len(final_data)}")

    train = []
    val = []

    for f in final_data:
        if random.random() < 0.1:
            val.append(f)
        else:
            train.append(f)

    print(f"Train: {len(train)}, Val: {len(val)}")

    save_to_json(f"dataset/TSP{mk}", "train", train)
    save_to_json(f"dataset/TSP{mk}", "val", val)

def get_path(folder) :
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', folder))

def save_to_json(folder, file_name, object):
    folder_path = get_path(folder)

    make_folder(folder_path)
    
    with open(os.path.join(folder_path, file_name + ".json"), 'w+') as outfile:
        json.dump(json.dumps(object), outfile)

def load_json(file_path):
    os_path = os.path.join(os.path.realpath(__file__), '..', file_path)
    with open(get_path(os_path), 'r') as json_file:
        data = json_file.read()
    return json.loads(data)

def load_all_json(folder, func=lambda x:x):
    return load_rec(get_path(folder), is_file_type(".json"), lambda x: func(load_single_json(x))) 

def load_all_file_paths(folder):
    return load_rec(get_path(folder), is_file_type(".qasm"), lambda x:x)   

def load_all_nx_graphs(folder):
    return load_rec(get_path(folder), is_file_type(".gml"), lambda x:load_nx_graph(x.path))  

def load_rec(file, func, load):
    files = []
    if not os.path.exists(file):
        return files
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

def save_jit_model(model, folder, file_name):
    path = get_path(folder)
    make_folder(path)

    for param in model.parameters():
        param.requires_grad = False

    torch.jit.script(model).save(os.path.join(path, file_name + ".pt"))

def load_model(path):
    torch.load(path)

def make_folder(folder_path):
    if not os.path.exists(folder_path):
        print("==> folder to save embedding does not exist... creating folder...")
        print("   ==> folder path: ", folder_path)
        os.mkdir(folder_path)