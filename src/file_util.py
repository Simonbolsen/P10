import json
import os
import networkx as nx
from tqdm import tqdm
import torch
import random

def split(folder):
    path = os.scandir(get_path(folder))
    for sf in tqdm(path):
        data = load_json(sf.path)
        file = {"data": data, "name": sf.name[:-5]}
        
        if(random.random() < 0.1):
            save_to_json("dataset/TSP2/val", file["name"], file)
        else:
            save_to_json("dataset/TSP2/train", file["name"], file)
        

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