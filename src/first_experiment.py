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

import matplotlib as mpl
mpl.use("TkAgg")


selected_algorithms = [
    #"ghz",
    #"graphstate",
    #"twolocalrandom",  # No good
    #"qftentangled", # Not working
    "dj",
    "qpeexact", # Not working
    "su2random",
    "wstate",
    "realamprandom"
]

# circuit_difficulty = {
#     "ghz": ,
#     "graphstate",
#     "twolocalrandom",
#     "qftentangled",
#     "dj",
#     "qpeexact", 
#     "su2random",
#     "wstate",
#     "realamprandom"
# }

# ---------------------- SUPPORT FUNCTIONS ------------------------------

def map_complex(data):
    if isinstance(data, list):
        return [map_complex(item) for item in data]
    else:
        return (data.real, data.imag)

def fast_contract_tdds(tdds, data, max_time=-1, max_node_size=-1):
    start_time = time.time()
    usable_path = data["path"]
    sizes = {i: [0, tdd.node_number()] for i, tdd in tdds.items()}

    for left_index, right_index in tqdm(usable_path):
        if max_time > 0 and int(time.time() - start_time) > max_time:
            data["conclusive"] = False
            print("Time limit for contraction reached. Aborting check")
            return None
        
        tdds[right_index] = cont(tdds[left_index], tdds[right_index])

        intermediate_tdd_size = tdds[right_index].node_number()
        sizes[right_index].append(intermediate_tdd_size)
        if max_node_size > 0 and intermediate_tdd_size > max_node_size:
            data["conclusive"] = False
            print("Node size limit reached. Aborting check")
            return None

    resulting_tdd = tdds[right_index]
    if "simulate" not in data or not data["simulate"]:
        data["equivalence"] = tddu.is_tdd_identitiy(resulting_tdd)
        data["conclusive"] = True
    else:
        data["equivalence"] = tddu.is_tdd_identitiy(resulting_tdd)
        data["conclusive"] = True
    data["sizes"] = sizes

    return resulting_tdd

def contract_tdds(tdds, data, max_time=-1, max_node_size=-1, save_intermediate_results=False, comprehensive_saving=False, folder_path=""):
    start_time = time.time()
    usable_path = data["path"]
    sizes = {i: [0, tdd.node_number()] for i, tdd in tdds.items()}
    folder = os.path.join(folder_path, "intermediate_results")
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    working_layer = ""
    it = 0 
    for left_index, right_index in tqdm(usable_path):
        if max_time > 0 and int(time.time() - start_time) > max_time:
            data["conclusive"] = False
            print("Time limit for contraction reached. Aborting check")
            return None
        if save_intermediate_results:
            if comprehensive_saving:
                working_layer = f"Contraction_{it}__{left_index}_{right_index}"
                if not os.path.exists(folder):
                    os.makedirs(folder, exist_ok=True)
                tdds[left_index].show(name=os.path.join(folder, working_layer, "tdd_" + str(left_index)))
                tdds[right_index].show(name=os.path.join(folder, working_layer, "tdd_" + str(right_index)))
                it += 1
        left_tensor = tddu.tensor_of_tdd(tdds[left_index])
        right_tensor = tddu.tensor_of_tdd(tdds[right_index])
        presumed_result_tensor = left_tensor.contract(right_tensor)

        tdds[right_index] = cont(tdds[left_index], tdds[right_index])

        result_tensor = tddu.tensor_of_tdd(tdds[right_index])
        presumed_result_tensor = presumed_result_tensor.transpose(*result_tensor.inds)
        # Check tdd and tensor are same
        same = presumed_result_tensor.inds == result_tensor.inds
        same = same and np.allclose(presumed_result_tensor.data, result_tensor.data)
        wrong_nodes = []
        if not same:
            if False and data["tdd_analysis"] is None:
                
                data["tdd_analysis"] = {
                    "left_tensor": map_complex(left_tensor.data.tolist()),
                    "right_tensor": map_complex(right_tensor.data.tolist()),
                    "result_tensor": map_complex(presumed_result_tensor.data.tolist()),
                    "actual_result_tensor": map_complex(result_tensor.data.tolist()),
                    "left_tensor_inds": left_tensor.inds,
                    "right_tensor_inds": right_tensor.inds,
                    "result_tensor_inds": presumed_result_tensor.inds,
                    "actual_result_tensor_inds": result_tensor.inds,
                    "tdds_in": os.path.join(folder, working_layer, "tdd_" + str(left_index)),
                    "left_tdd_name": left_index,
                    "right_tdd_name": right_index
                }

            data["not_same_tensors"].append((left_index, right_index))
            wrong_nodes.append(right_index)
        if False and same and len(left_tensor.inds) == 6 and len(right_tensor.inds) == 4:
            if data["correct_example"] is None:
                
                data["correct_example"] = {
                    "left_tensor": map_complex(left_tensor.data.tolist()),
                    "right_tensor": map_complex(right_tensor.data.tolist()),
                    "result_tensor": map_complex(presumed_result_tensor.data.tolist()),
                    "actual_result_tensor": map_complex(result_tensor.data.tolist()),
                    "left_tensor_inds": left_tensor.inds,
                    "right_tensor_inds": right_tensor.inds,
                    "result_tensor_inds": presumed_result_tensor.inds,
                    "actual_result_tensor_inds": result_tensor.inds,
                    "tdds_in": os.path.join(folder, working_layer, "tdd_" + str(left_index)),
                    "left_tdd_name": left_index,
                    "right_tdd_name": right_index
                }
        data["path_data"]["dot"] = tnu.get_dot_from_path(usable_path, wrong_nodes)

        intermediate_tdd_size = tdds[right_index].node_number()
        sizes[right_index].append(intermediate_tdd_size)
        if save_intermediate_results:
            file_path = os.path.join(folder, working_layer, "tdd_" + str(left_index) + "_" + str(right_index))
            tdds[right_index].show(name=file_path)
        if max_node_size > 0 and intermediate_tdd_size > max_node_size:
            data["conclusive"] = False
            print("Node size limit reached. Aborting check")
            return None

    resulting_tdd = tdds[right_index]
    if "simulate" not in data or not data["simulate"]:
        data["equivalence"] = tddu.is_tdd_identitiy(resulting_tdd)
        data["conclusive"] = True
    else:
        data["equivalence"] = tddu.is_tdd_identitiy(resulting_tdd)
        data["conclusive"] = True
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
    experiment_name = f"mapping_experiment_{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
    folder_path = os.path.join("experiments", experiment_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    # Experiment settings
    settings = {
        "simulate": False,
        "algorithms": selected_algorithms,
        "levels": [(0, 2)], #[i for i in combinations(range(4), 2)],
        "qubits": [i for i in range(4, 257, 2)]
    }

    settings = {
        "simulate": False,
        "algorithms": ["dj"],#, "ghz", "graphstate", "qftentangled"],
        "levels": [(0, 2)],
        "qubits": range(208,257)#sorted(list(set([int(x**(3/2)) for x in range(2, 41)])))#list(set([int(2**(x/4)) for x in range(4, 30)]))
    }

    print(f"Performing experiment with {settings['algorithms']} for levels: {settings['levels']}\n\tqubits: {settings['qubits']}")

    # Prepare benchmark circuits:
    circuit_configs = get_all_configs(settings)

    # For each circuit, run equivalence checking:
    for circ_conf in circuit_configs:
        circ_conf["random_gate_deletions"] = 0
        # Prepare data container
        data = {
            #"max_rank": circuit_difficulty[circ_conf["algorithm"]] * circ_conf["qubits"],
            "experiment_name": experiment_name,
            "file_name": f"circuit_{circ_conf['algorithm']}_{circ_conf['level'][0]}{circ_conf['level'][1]}_{circ_conf['qubits']}",
            "contraction_settings": {
                "max_time": -1, # in seconds, -1 for inf
                "max_replans": 3,
                "max_intermediate_node_size": -1 #-1 for inf
            },
            "circuit_settings": circ_conf,
            "circuit_data": {},
            "path_settings": {
                "method": "cotengra",
                "opt_method": "greedy", # greedy, betweenness, walktrap
                "minimize": "flops",
                "max_repeats": 50,
                "max_time": 60
            },
            "path_data": {},
            "not_same_tensors": [],
            "tdd_analysis": None,
            "correct_example": None
        }

        if "simulate" in settings and settings["simulate"]:
            options = [[1 + 0j, 0j], [0j, 1 + 0j]]
            settings["state"] = [random.choice(options) for _ in range(circ_conf["qubits"])]
        
        working_path = os.path.join(folder_path, data["file_name"])
        if not os.path.exists(working_path):
            os.makedirs(working_path, exist_ok=True)

        # Prepare circuit
        print("Preparing circuits...")
        #circuit = bu.get_circuit_setup_quimb(bu.get_benchmark_circuit(circ_conf), draw=False)
        starting_time = time.time_ns()
        circuit = bu.get_dual_circuit_setup_quimb(data, draw=False)
        data["circuit_setup_time"] = int((time.time_ns() - starting_time) / 1000000)

        # Transform to tensor networks (without initial states and cnot decomp)
        print("Constructing tensor network...")
        starting_time = time.time_ns()
        tensor_network = tnu.get_tensor_network(circuit, split_cnot=False, 
                                                state = settings["state"] if "state" in settings else None)
        data["tn_construnction_time"] = int((time.time_ns() - starting_time) / 1000000)
        
        #tensor_network.draw(color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'])
        print(f"Number of tensors: {len(tensor_network.tensor_map)}")

        attempts = 0
        succeeded = False
        while (attempts < data["contraction_settings"]["max_replans"] and not succeeded):
            attempts += 1

            # Construct the plan from CoTenGra
            print("Find contraction path...")
            starting_time = time.time_ns()
            path = tnu.get_contraction_path(tensor_network, data)
            data["path_construction_time"] = int((time.time_ns() - starting_time) / 1000000)

            #tn_draw.draw_tn(tensor_network, color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'], save_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
            #tnu.draw_contraction_order(tensor_network, path, save_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))

            # Prepare gate TDDs
            print("Preparing gate TDDs...")
            starting_time = time.time_ns()
            gate_tdds = tddu.get_tdds_from_quimb_tensor_network(tensor_network)
            data["gate_prep_time"] = int((time.time_ns() - starting_time) / 1000000)

            #tddu.draw_all_tdds(gate_tdds, folder=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
            starting_time = time.time_ns()
            data["qcec_equivalence"] = verify(data["circuit_data"]["circuit_1_qasm"], data["circuit_data"]["circuit_2_qasm"]).equivalence.value in [1,4,5]  # see https://mqt.readthedocs.io/projects/qcec/en/latest/library/EquivalenceCriterion.html
            data["qcec_time"] = int((time.time_ns() - starting_time) / 1000000)
            print(f"QCEC says: {data['qcec_equivalence']}")
            data["circuit_data"]["circuit_1_qasm"] = data["circuit_data"]["circuit_1_qasm"].qasm()
            data["circuit_data"]["circuit_2_qasm"] = data["circuit_data"]["circuit_2_qasm"].qasm()
            # quimb_result = tensor_network.contract(optimize=data["path_data"]["original_path"])
            # variable_order = sorted(list(quimb_result.inds), key=reverse_lexicographic_key, reverse=True)
            # processed_result = quimb_result.transpose(*variable_order, inplace=False)
            # quimb_result_tdd = Tensor(processed_result.data, [Index(s) for s in processed_result.inds]).tdd()
            # quimb_result_tdd.show(name=os.path.join(working_path, data["file_name"] + f"_R{attempts}" + "_tensor_cont"))
            # data["quimb_equivalence"] = tddu.is_tdd_identitiy(quimb_result_tdd)

            # np.array([v.real if abs(v) > 0.01 else 0 for v in (quimb_result.data*(-1j)).flatten()]).reshape((32,32))

            # Contract TDDs + equivalence checking
            print(f"Contracting {len(path)} times...")
            starting_time = time.time_ns()
            #result_tdd = contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"], save_intermediate_results=False, comprehensive_saving=False, folder_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
            result_tdd = fast_contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"])
            data["contraction_time"] = int((time.time_ns() - starting_time) / 1000000)

            if (data["qcec_equivalence"] != data["equivalence"]):
                print('\033[31m' + "Erroneous result: Quimb != TDD" + '\033[m')

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

        # ------------------------------- RUNNING THE LINEAR PLAN -------------------------------
        if False:
            print("------------------- Starting Linear Run -------------------")
            data["path_settings"]["method"] = "linear"
            data["linear_fraction"] = 0
            data["file_name"] = data["file_name"] + "_lin"

            print("Find contraction path...")
            starting_time = time.time_ns()
            path = tnu.get_contraction_path(tensor_network, data)
            data["path_construction_time"] = int((time.time_ns() - starting_time) / 1000000)

            tn_draw.draw_tn(tensor_network, color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'], save_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
            #tnu.draw_contraction_order(tensor_network, path, save_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))

            # Prepare gate TDDs
            print("Preparing gate TDDs...")
            starting_time = time.time_ns()
            gate_tdds = tddu.get_tdds_from_quimb_tensor_network(tensor_network)
            data["gate_prep_time"] = int((time.time_ns() - starting_time) / 1000000)

            #tddu.draw_all_tdds(gate_tdds, folder=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))

            # Contract TDDs + equivalence checking
            print(f"Contracting {len(path)} times...")
            starting_time = time.time_ns()
            result_tdd = contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"], save_intermediate_results=False, comprehensive_saving=False, folder_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
            data["contraction_time"] = int((time.time_ns() - starting_time) / 1000000)

            # Save data for circuit
            if not debug:
                print("Saving data...")
                file_path = os.path.join(working_path, data["file_name"] + f"_R{attempts}" + ".json")
                with open(file_path, "w") as file:
                    json.dump(data, file, indent=4)

            result_tdd.show(name=os.path.join(working_path, data["file_name"] + f"_R{attempts}" + "_TDD"))
            if result_tdd is not None:
                succeeded = True
            else:
                print(f"Retry #{attempts+1}")

            if not succeeded:
                print("Failed to check equivalence of circuits")

            #print("Saving resulting TDD image...")
            #result_tdd.show(name="tester")

    # Save collected data


if __name__ == "__main__":
    first_experiment()
