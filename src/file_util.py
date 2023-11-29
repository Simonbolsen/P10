import json
import os

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

"""Data that needs to be collected:

From circuits:
 - Number of gates
 - Number of qubits
 - Dimensionality of gates (or ordering of gates by dimensionality)
 - Number of unique gates used
 - Number of custom gates initially

From tensor networks:
 - Number pf edges between tensors (shared indices)
 - Number of tensors (and dimensionalities)


From TDDs:

From contraction plan:
- The plan

From each performed contraction:



Other:




"""


