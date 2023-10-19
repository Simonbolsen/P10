from quimb.tensor import Circuit
import cotengra as ctg
import numpy as np
import random
from tddpure.TDD.TDD import Ini_TDD, TDD
from tddpure.TDD.TN import Index,Tensor,TensorNetwork
from tddpure.TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs
import circuit_util as cu
import tdd_util as tddu
import tensor_network_util as tnu
import bench_util as bu
import os
from datetime import datetime
import time
import json
from tqdm import tqdm
from itertools import combinations

import matplotlib as mpl
mpl.use("TkAgg")


selected_algorithms = [
    "ghz",
    "graphstate",
    "twolocalrandom",
    "qftentangled", # Not working
    "dj",
    "qpeexact", # Not working
    "su2random",
    "wstate",
    "realamprandom"
]

# ---------------------- SUPPORT FUNCTIONS ------------------------------
def contract_tdds(tdds, data):
    usable_path = data["path"]
    sizes = {i: [0, tdd.node_number()] for i, tdd in tdds.items()}

    for left_index, right_index in tqdm(usable_path):
        tdds[right_index] = tddu.cont(tdds[left_index], tdds[right_index])
        sizes[right_index].append(tdds[right_index].node_number())

    resulting_tdd = tdds[right_index]
    data["equivalence"] = tddu.is_tdd_identitiy(resulting_tdd)
    data["sizes"] = sizes

    return resulting_tdd

def get_all_configs(settings):
    all_configs = []
    for algorithm in settings["algorithms"]:
        for level in settings["levels"]:
            for qubit in settings["qubits"]:
                all_configs.append({"algorithm": algorithm, "level": level, "qubits": qubit})

    return all_configs

debug=False
def first_experiment():
    # Prepare save folder and file paths
    experiment_name = f"first_experiment_{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
    folder_path = os.path.join("experiments", experiment_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    # Experiment settings
    settings = {
        "algorithms": selected_algorithms,
        "levels": [i for i in combinations(range(4), 2)],
        "qubits": range(2, 10)
    }

    settings = {
        "algorithms": ["dj"],
        "levels": [(0, 2)],
        "qubits": sorted(list(set([int(x**(3/2)) for x in range(2, 41)])))#list(set([int(2**(x/4)) for x in range(4, 30)]))
    }

    print(f"Performing experiment with {settings['algorithms']} for levels: {settings['levels']}\n\tqubits: {settings['qubits']}")

    # Prepare benchmark circuits:
    circuit_configs = get_all_configs(settings)

    # For each circuit, run equivalence checking:
    for circ_conf in circuit_configs:
        # Prepare data container
        data = {
            "experiment_name": experiment_name,
            "file_name": f"circuit_{circ_conf['algorithm']}_{circ_conf['level'][0]}{circ_conf['level'][1]}_{circ_conf['qubits']}",
            "circuit_settings": circ_conf,
            "circuit_data": {},
            "path_settings": {
                "method": "cotengra",
                "opt_method": "greedy",
                "minimize": "flops",
                "max_repeats": 80,
                "max_time": 10
            },
            "path_data": {}
        }

        # Prepare circuit
        print("Preparing circuits...")
        #circuit = bu.get_circuit_setup_quimb(bu.get_benchmark_circuit(circ_conf), draw=False)
        starting_time = time.time_ns()
        circuit = bu.get_dual_circuit_setup_quimb(data, draw=False)
        data["circuit_setup_time"] = int((time.time_ns() - starting_time) / 1000000)

        # Transform to tensor networks (without initial states and cnot decomp)
        print("Constructing tensor network...")
        starting_time = time.time_ns()
        tensor_network = tnu.get_tensor_network(circuit, include_state = False, split_cnot=False)
        data["tn_construnction_time"] = int((time.time_ns() - starting_time) / 1000000)
        
        #tensor_network.draw(color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'])
        print(f"Number of tensors: {len(tensor_network.tensor_map)}")

        # Construct the plan from CoTenGra
        print("Find contraction path...")
        starting_time = time.time_ns()
        path = tnu.get_contraction_path(tensor_network, data)
        data["path_construction_time"] = int((time.time_ns() - starting_time) / 1000000)
        

        # Prepare gate TDDs
        print("Preparing gate TDDs...")
        starting_time = time.time_ns()
        gate_tdds = tddu.get_tdds_from_quimb_tensor_network(tensor_network)
        data["gate_prep_time"] = int((time.time_ns() - starting_time) / 1000000)
        

        # Contract TDDs + equivalence checking
        print(f"Contracting {len(path)} times...")
        starting_time = time.time_ns()
        result_tdd = contract_tdds(gate_tdds, data)
        data["contraction_time"] = int((time.time_ns() - starting_time) / 1000000)

        #print("Saving resulting TDD image...")
        #result_tdd.show(name="tester")

        # Save data for circuit
        if not debug:
            print("Saving data...")
            file_path = os.path.join(folder_path, data["file_name"] + ".json")
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)

    # Save collected data


if __name__ == "__main__":
    first_experiment()
