import json
import os

def load_json(file_path):
    os_path = os.path.join(os.path.realpath(__file__), '..', file_path)
    with open(os_path, 'r') as json_file:
        data = json_file.read()
    return json.loads(data)

def load_all_json(folder):
    path = os.scandir(path=os.path.join(os.path.realpath(__file__), '..', '..', folder))
    files = []

    for file in path:
        if not file.is_dir() and os.path.splitext(file)[1] == ".json":
            with open(file, 'r') as json_file:
                data = json_file.read()
                files.append(json.loads(data))

    return files

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


