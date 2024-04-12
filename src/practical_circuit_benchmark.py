from quimb.tensor import Circuit
import cotengra as ctg
import numpy as np
import random
from tddpure.TDD.TDD import Ini_TDD, TDD, tdd_2_np, cont
from tddpure.TDD.TN import Index,Tensor,TensorNetwork
from tddpure.TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs
import circuit_util as cu
import tdd_util as tddu
from tdd_util import reverse_lexicographic_key
import tensor_network_util as tnu
import bench_util as bu
import os
from datetime import datetime
import time
import json
import tn_draw
from tqdm import tqdm
from itertools import combinations
from mqt.qcec import verify
from copy import deepcopy
from contraction_experiments import *
import file_util as fu

import sys
sys.setrecursionlimit(2000)

import matplotlib as mpl
mpl.use("TkAgg")







def get_all_circuit_files(folder_name):
    actual_folder = os.path.join(folder_name)
    return fu.load_all_file_paths(actual_folder)

def experiment():
    # Prepare save folder and file paths
    experiment_name = f"practical_circuits_benchmark_{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
    folder_path = os.path.join("experiments", experiment_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    files = get_all_circuit_files("qasm_circuits")
    

    # For each circuit, run equivalence checking:
    for file in files:
        #circ_conf["random_gate_deletions"] = 3
        print(f"Running algorithm: {file.name.replace('.qasm', '')}")
        # Prepare data container
        circ_conf = {
                "simulate": True,
                "level": (0, 2),
                "random_gate_deletions": 0,
                "repetition": 1,
                "sliced": False,
                "cnot_split": False,
                "use_subnets": False
        }
        circ_conf["algorithm"] = file.name.replace(".qasm", "")
        circ_conf["algorithm_file_path"] = file.path

        data = {
            "version": 1,
            "experiment_name": experiment_name,
            "expect_equivalence": circ_conf['random_gate_deletions'] == 0,
            "file_name": f"circuit_{circ_conf['algorithm']}_{circ_conf['level'][0]}{circ_conf['level'][1]}_d{circ_conf['random_gate_deletions']}_r{circ_conf['repetition']}",
            "settings": circ_conf,
            "contraction_settings": {
                "max_time": 300, # in seconds, -1 for inf
                "max_replans": 1,
                "max_intermediate_node_size": -1 #-1 for inf
            },
            "circuit_settings": circ_conf,
            "circuit_data": {},
            "path_settings": {
                "method": "cotengra",
                "opt_method": "rgreedy", #  kahypar-balanced, kahypar-agglom, labels, labels-agglom
                "minimize": "flops",
                "max_repeats": 50,
                "max_time": 60,
                "use_proportional": True,
                "gridded": False,
                "linear_fraction": 0
            },
            "path_data": {},
            "not_same_tensors": [],
            "tdd_analysis": None,
            "correct_example": None
        }

        working_path = os.path.join(folder_path, data["file_name"])
        if not os.path.exists(working_path):
            os.makedirs(working_path, exist_ok=True)

        # Prepare circuit
        print("Preparing circuits...")
        #circuit = bu.get_circuit_setup_quimb(bu.get_benchmark_circuit(circ_conf), draw=False)
        starting_time = time.time_ns()
        circuit = bu.get_dual_circuit_setup_from_practical_circuits(data, draw=False)
        data["circuit_setup_time"] = int((time.time_ns() - starting_time) / 1000000)
        data["circuit_settings"]["qubits"] = circuit.N

        if "simulate" in circ_conf and circ_conf["simulate"]:
            options = [[1 + 0j, 0j], [0j, 1 + 0j]]
            circ_conf["state"] = [random.choice(options) for _ in range(circ_conf["qubits"])]

        # Transform to tensor networks (without initial states and cnot decomp)
        print("Constructing tensor network...")
        starting_time = time.time_ns()
        tensor_network = tnu.get_tensor_network(circuit, split_cnot=circ_conf["cnot_split"], 
                                                state = circ_conf["state"] if "state" in circ_conf else None)
        if circ_conf["sliced"]:
            tnu.slice_tensor_network_vertically(tensor_network)
        data["tn_construnction_time"] = int((time.time_ns() - starting_time) / 1000000)
        
        #tensor_network.draw(color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'])
        print(f"Number of tensors: {len(tensor_network.tensor_map)}")

        if data["settings"]["use_subnets"]:
            print(f"Attemping tensor network split...")
            sub_tensor_networks = tnu.find_and_split_subgraphs_in_tn(tensor_network)
            print(f"Split tensor network into {len(sub_tensor_networks)}")
            data["sub_networks"] = len(sub_tensor_networks)
        else:
            sub_tensor_networks = [tensor_network]

        attempts = 0
        succeeded = False
        while (attempts < data["contraction_settings"]["max_replans"] and not succeeded):
            attempts += 1

            # print("Starting QCEC sanity check")
            # starting_time = time.time_ns()
            # data["qcec_equivalence"] = verify(data["circuit_data"]["circuit_1_qasm"], data["circuit_data"]["circuit_2_qasm"]).equivalence.value in [1,4,5]  # see https://mqt.readthedocs.io/projects/qcec/en/latest/library/EquivalenceCriterion.html
            # data["qcec_time"] = int((time.time_ns() - starting_time) / 1000000)
            # print(f"QCEC says: {data['qcec_equivalence']}")
            data["circuit_data"]["circuit_1_qasm"] = data["circuit_data"]["circuit_1_qasm"].qasm()
            data["circuit_data"]["circuit_2_qasm"] = data["circuit_data"]["circuit_2_qasm"].qasm()

            # quimb_result = tensor_network.contract(optimize=data["path_data"]["original_path"])
            # variable_order = sorted(list(quimb_result.inds), key=reverse_lexicographic_key, reverse=True)
            # processed_result = quimb_result.transpose(*variable_order, inplace=False)
            # quimb_result_tdd = Tensor(processed_result.data, [Index(s) for s in processed_result.inds]).tdd()
            # quimb_result_tdd.show(name=os.path.join(working_path, data["file_name"] + f"_R{attempts}" + "_tensor_cont"))
            # data["quimb_equivalence"] = tddu.is_tdd_identitiy(quimb_result_tdd)
            # print(f"Quimb says: {data['quimb_equivalence']}")
            # np.array([v.real if abs(v) > 0.01 else 0 for v in (quimb_result.data*(-1j)).flatten()]).reshape((32,32))

            data_containers = [deepcopy(data) for _ in sub_tensor_networks]
            for i, stn in enumerate(sub_tensor_networks):
                data = data_containers[i]

                # Construct the plan from CoTenGra
                print("Find contraction path...")
                starting_time = time.time_ns()
                path = tnu.get_contraction_path(stn, data)
                data["path_construction_time"] = int((time.time_ns() - starting_time) / 1000000)

                #tn_draw.draw_tn(tensor_network, color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'], save_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
                #tnu.draw_contraction_order(tensor_network, path, save_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))

                # Prepare gate TDDs
                print("Preparing gate TDDs...")
                starting_time = time.time_ns()
                gate_tdds = tddu.get_tdds_from_quimb_tensor_network(stn)
                data["gate_prep_time"] = int((time.time_ns() - starting_time) / 1000000)

                #tddu.draw_all_tdds(gate_tdds, folder=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
                

                # Contract TDDs + equivalence checking
                print(f"Contracting {len(path)} times...")
                starting_time = time.time_ns()
                #result_tdd = contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"], save_intermediate_results=True, comprehensive_saving=True, folder_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
                result_tdd = fast_contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"])
                data["contraction_time"] = int((time.time_ns() - starting_time) / 1000000)

                data_containers[i] = data

            data = combine_data_containers(data_containers)

            if (data["expect_equivalence"] != data["equivalence"]):
                print('\033[31m' + "Erroneous result: Expected != TDD" + '\033[m')
                return

            # Save data for circuit
            if not debug:
                print("Saving data...")
                file_path = os.path.join(working_path, data["file_name"] + f"_R{attempts}" + ".json")
                with open(file_path, "w") as file:
                    json.dump(data, file, indent=4)

            #result_tdd.show(name=os.path.join(working_path, data["file_name"] + f"_R{attempts}" + "_TDD"))
            if result_tdd is not None:
                succeeded = True
            else:
                print(f"Retry #{attempts+1}")

if __name__ == "__main__":
    experiment()