import json
import os
import networkx as nx

def load_json(file_path):
    os_path = os.path.join(os.path.realpath(__file__), '..', file_path)
    with open(os_path, 'r') as json_file:
        data = json_file.read()
    return json.loads(data)

def load_all_json(folder):
    files = []
    load_rec_json(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', folder)), files)    
    return files

def load_all_file_paths(folder):
    files = []
    load_rec_paths(os.path.normpath(os.path.join(os.path.dirname(__file__), '..', folder)), files)    
    return files

def load_rec_paths(file, files):
    path = os.scandir(file)
    for sf in path:
        if sf.is_dir():
            load_rec_json(sf, files)
        elif is_qasm_file(sf):
            files.append(sf)

def load_rec_json(file, files):
    path = os.scandir(file)
    for sf in path:
        if sf.is_dir():
            load_rec_json(sf, files)
        elif is_json(sf):
            files.append(load_single_json(sf))

def is_json(file):
    return os.path.splitext(file)[1] == ".json"

def is_qasm_file(file):
    return os.path.splitext(file)[1] == ".qasm"

def load_single_json(file):
    with open(file, 'r') as json_file:
        return json.loads(json_file.read())

def save_nx_graph(graph: nx.Graph, path: str, file_name: str):
    final_path = file_name + ".gml" if path == "" else os.path.join(path, file_name + ".gml")
    nx.write_gml(graph, final_path)

def load_nx_graph(path: str):
    graph = nx.read_gml(path)
    return graph

