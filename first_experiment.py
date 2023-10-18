from quimb.tensor import Circuit
import cotengra as ctg
import numpy as np
import random
from tddfork.TDD.TDD import Ini_TDD, TDD
from tddfork.TDD.TN import Index,Tensor,TensorNetwork
from tddfork.TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs
import circuit_util as cu
import tdd_util as tddu
import tensor_network_util as tnu
import bench_util as bu
import os
from datetime import datetime
import json

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

    for left_index, right_index in usable_path:
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

debug=True
def first_experiment():
    # Prepare save folder and file paths
    folder_path = os.path.join("experiments", f"first_experiment_{datetime.today().strftime('%Y-%m-%d')}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    # Experiment settings
    settings = {
        "algorithms": selected_algorithms,
        "levels": range(4),
        "qubits": range(2, 10)
    }

    settings = {
        "algorithms": ["dj"],
        "levels": [0],
        "qubits": [4]
    }

    # Prepare benchmark circuits:
    circuit_configs = get_all_configs(settings)

    # For each circuit, run equivalence checking:
    for circ_conf in circuit_configs:
        # Prepare data container
        data = {
            "circuit_settings": circ_conf,
            "path_settings": {
                "method": "cotengra",
                "minimize": "flops",
                "max_repeats": 128,
                "max_time": 60
            }
        }

        # Prepare circuit
        circuit = bu.get_circuit_setup_quimb(bu.get_benchmark_circuit(circ_conf), draw=True)

        # Transform to tensor networks (without initial states and cnot decomp)
        tensor_network = tnu.get_tensor_network(circuit, include_state = False, split_cnot=False)
        tensor_network.draw(color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'])

        # Construct the plan from CoTenGra
        path = tnu.get_contraction_path(tensor_network, data["path_settings"])
        data["path"] = path

        # Prepare gate TDDs
        gate_tdds = tddu.get_tdds_from_quimb_tensor_network(tensor_network)

        # Contract TDDs + equivalence checking
        result_tdd = contract_tdds(gate_tdds, data)
        result_tdd.show(name="tester")

        # Save data for circuit
        if not debug:
            file_path = os.path.join(folder_path, f"circuit_{circ_conf['algorithm']}_{circ_conf['level']}_{circ_conf['qubits']}.json")
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)

    # Save collected data


if __name__ == "__main__":
    first_experiment()
