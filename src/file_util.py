import json
import os
import networkx as nx

def get_path(folder) :
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', folder))

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
    return load_rec(get_path(folder), is_file_type(".gml"), lambda x:load_nx_graph(x))  

def load_rec(file, func, load):
    files = []
    path = os.scandir(file)
    for sf in path:
        if sf.is_dir():
            files += load_rec(sf, files, func, load)
        elif func(sf):
            files.append(load(sf))

    return files

def is_file_type(ftype):
    return lambda file: os.path.splitext(file)[1] == ftype

def load_single_json(file):
    with open(file, 'r') as json_file:
        return json.loads(json_file.read())

def save_nx_graph(graph: nx.Graph, path: str, file_name: str):
    final_path = file_name + ".gml" if path == "" else os.path.join(path, file_name + ".gml")
    nx.write_gml(graph, final_path)

def load_nx_graph(path: str):
    graph = nx.read_gml(path)
    return graph