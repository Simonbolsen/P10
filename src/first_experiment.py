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
from graph_nn import EdgePredictionGNN

import sys
sys.setrecursionlimit(2000)

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

    if data["make_dataset"]:
        data["contractions"] = []

    for left_index, right_index in tqdm(usable_path):
        if max_time > 0 and int(time.time() - start_time) > max_time:
            data["failed"] = True
            data["conclusive"] = False
            print("Time limit for contraction reached. Aborting check")
            return None
        
        current_data = {
            "left": {},
            "right": {},
            "result": {}
        }

        if data["make_dataset"]:
            current_data["left"]["nodes"] = tdds[left_index].node_number()
            current_data["left"]["indices"] = [index.name for index in tdds[left_index].index_set]
            current_data["left"]["gates"] = tdds[left_index].gates
            current_data["right"]["nodes"] = tdds[right_index].node_number()
            current_data["right"]["indices"] = [index.name for index in tdds[right_index].index_set]
            current_data["right"]["gates"] = tdds[right_index].gates
            start_time = time.time_ns()

        tdds[right_index] = cont(tdds[left_index], tdds[right_index])

        if data["make_dataset"]:
            current_data["result"]["nodes"] = tdds[right_index].node_number()
            current_data["result"]["indices"] = [index.name for index in tdds[right_index].index_set]
            current_data["result"]["gates"] = tdds[right_index].gates
            current_data["result"]["time"] = int((time.time_ns() - start_time) / 1000000)
            data["contractions"].append(current_data)

        intermediate_tdd_size = tdds[right_index].node_number()
        sizes[right_index].append(intermediate_tdd_size)
        if max_node_size > 0 and intermediate_tdd_size > max_node_size:
            data["failed"] = True
            data["conclusive"] = False
            print("Node size limit reached. Aborting check")
            return None

    resulting_tdd = tdds[right_index]
    if "simulate" not in data["settings"] or not data["settings"]["simulate"]:
        data["equivalence"] = tddu.is_tdd_identitiy(resulting_tdd, data["circuit_settings"]["qubits"] if "sub_networks" not in data or data["sub_networks"] == 1 else -1)
        data["conclusive"] = True
    else:
        data["equivalence"] = tddu.is_tdd_equal(resulting_tdd, data["settings"]["state"])
        data["conclusive"] = not data["equivalence"]
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
            if data["tdd_analysis"] is None:
                
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
        if same and len(left_tensor.inds) == 6 and len(right_tensor.inds) == 4:
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
    if "simulate" not in data["settings"] or not data["settings"]["simulate"]:
        data["equivalence"] = tddu.is_tdd_identitiy(resulting_tdd, data["circuit_settings"]["qubits"])
        data["conclusive"] = True
    else:
        data["equivalence"] = tddu.is_tdd_equal(resulting_tdd, data["state"])
        data["conclusive"] = not data["equivalence"]
    data["sizes"] = sizes

    return resulting_tdd

def get_all_configs(settings):
    all_configs = []
    for algorithm in settings["algorithms"]:
        for level in settings["levels"]:
            for qubit in settings["qubits"]:
                for dels in settings["random_gate_dels_range"]:
                    for i in range(settings["repetitions"]):
                        all_configs.append({"algorithm": algorithm, "level": level, "qubits": qubit, "random_gate_deletions": dels, "repetition": i})

    return all_configs

debug=False
def first_experiment(iter_settings, settings, contraction_settings, path_settings, folder_name="garbage", folder_with_time=True):
    # Prepare save folder and file paths
    experiment_name = f"{folder_name}_{datetime.today().strftime('%Y-%m-%d_%H-%M') if folder_with_time else ''}"
    folder_path = os.path.join("experiments", experiment_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    # Experiment settings     
    prev_rep = 0
    circuit_configs = get_all_configs(iter_settings)
    settings = settings | iter_settings

    # For each circuit, run equivalence checking:
    for circ_conf in circuit_configs:
        # Prepare data container
        data = {
            "version": 2,
            "experiment_name": experiment_name,
            "expect_equivalence": circ_conf['random_gate_deletions'] == 0,
            "file_name": f"circuit_{circ_conf['algorithm']}_{circ_conf['level'][0]}{circ_conf['level'][1]}_{circ_conf['qubits']}_d{circ_conf['random_gate_deletions']}_r{circ_conf['repetition']+prev_rep}",
            "settings": settings,
            "contraction_settings": contraction_settings,
            "circuit_settings": circ_conf,
            "circuit_data": {},
            "path_settings": path_settings,
            "path_data": {},
            "not_same_tensors": [],
            "tdd_analysis": None,
            "correct_example": None,
            "failed": False,
            "make_dataset": True
        }

        if "simulate" in settings and settings["simulate"]:
            options = [[1 + 0j, 0j], [0j, 1 + 0j]]
            settings["state"] = [random.choice(options) for _ in range(circ_conf["qubits"])]
        
        working_path = os.path.join(folder_path, data["file_name"])
        if not os.path.exists(working_path):
            os.makedirs(working_path, exist_ok=True)

        if data["make_dataset"]:
            dataset_folder_path = os.path.join("dataset", "tdd_size_predict")
            data_file_name = f"datapoint_{circ_conf['algorithm']}_{'sim' if settings['simulate'] else 'equiv'}_{circ_conf['qubits']}_r{circ_conf['repetition']+prev_rep}"
            dataset_file_path = os.path.join(dataset_folder_path, data_file_name + ".json")
            if os.path.exists(dataset_file_path):
                print(f"Skipping: {dataset_file_path}")
                continue

        # Prepare circuit
        print("Preparing circuits...")
        #circuit = bu.get_circuit_setup_quimb(bu.get_benchmark_circuit(circ_conf), draw=False)
        starting_time = time.time_ns()
        circuit = bu.get_dual_circuit_setup_quimb(data, draw=False) if data["circuit_settings"]["algorithm"] != "random" else bu.get_gauss_random_circuit(data["circuit_settings"]["qubits"])
        data["circuit_setup_time"] = int((time.time_ns() - starting_time) / 1000000)

        if data["settings"]["use_qcec_only"]:
            print("Starting QCEC sanity check")
            starting_time = time.time_ns()
            data["equivalence"] = verify(data["circuit_data"]["circuit_1_qasm"], data["circuit_data"]["circuit_2_qasm"]).equivalence.value in [1,4,5]  # see https://mqt.readthedocs.io/projects/qcec/en/latest/library/EquivalenceCriterion.html
            data["qcec_time"] = int((time.time_ns() - starting_time) / 1000000)
            print(f"QCEC says: {data['equivalence']}")

            data["circuit_data"]["circuit_1_qasm"] = data["circuit_data"]["circuit_1_qasm"].qasm()
            data["circuit_data"]["circuit_2_qasm"] = data["circuit_data"]["circuit_2_qasm"].qasm()

            print("Saving data...")
            file_path = os.path.join(working_path, data["file_name"] + ".json")
            with open(file_path, "w") as file:
                json.dump(data, file, indent=4)
            continue

        # Transform to tensor networks (without initial states and cnot decomp)
        print("Constructing tensor network...")
        starting_time = time.time_ns()
        tensor_network = tnu.get_tensor_network(circuit, split_cnot=settings["cnot_split"], 
                                                state = settings["state"] if "state" in settings else None)
        if settings["sliced"]:
            tnu.slice_tensor_network_vertically(tensor_network)
        data["tn_construnction_time"] = int((time.time_ns() - starting_time) / 1000000)
        
        #tensor_network.draw(color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'])
        print(f"Number of tensors: {len(tensor_network.tensor_map)}")

        variable_order = sorted(list(tensor_network.all_inds()), key=reverse_lexicographic_key, reverse=True)
        Ini_TDD(variable_order, max_rank=len(variable_order)+1)
        print(f"Using rank {len(variable_order)+1} for TDDs")

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
            if bool(data["circuit_data"]):
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

            resulting_sub_tdds = []
            data_containers = [deepcopy(data) for _ in sub_tensor_networks]
            for i, stn in enumerate(sub_tensor_networks):
                data = data_containers[i]

                # Construct the plan from CoTenGra
                print("Find contraction path...")
                starting_time = time.time_ns()
                path = tnu.get_contraction_path(stn, circuit, data)
                data["path_construction_time"] = int((time.time_ns() - starting_time) / 1000000)

                #tn_draw.draw_tn(tensor_network, color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'], save_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
                #tnu.draw_contraction_order(tensor_network, path, save_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))

                # Prepare gate TDDs
                print("Preparing gate TDDs...")
                starting_time = time.time_ns()
                gate_tdds = tddu.get_tdds_from_quimb_tensor_network(stn, with_input=settings["simulate"])
                data["gate_prep_time"] = int((time.time_ns() - starting_time) / 1000000)

                #tddu.draw_all_tdds(gate_tdds, folder=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
                

                # Contract TDDs + equivalence checking
                print(f"Contracting {len(path)} times...")
                starting_time = time.time_ns()
                #result_tdd = contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"], save_intermediate_results=True, comprehensive_saving=True, folder_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
                result_tdd = fast_contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"])
                data["contraction_time"] = int((time.time_ns() - starting_time) / 1000000)

                data_containers[i] = data

                if len(sub_tensor_networks) > 1:
                    resulting_sub_tdds.append(result_tdd)

            if any([data_containers[i]["failed"] for i in range(len(data_containers))]):
                continue
            
            if data["make_dataset"]:
                all_contractions = []
                for dc in data_containers:
                    all_contractions.extend(dc["contractions"]) 
                    dc["contractions"] = []
                # Save data 
                dataset_folder_path = os.path.join("dataset", "tdd_size_predict")
                if not os.path.exists(dataset_folder_path):
                    os.makedirs(dataset_folder_path, exist_ok=True)
                data_file_name = f"datapoint_{circ_conf['algorithm']}_{'sim' if settings['simulate'] else 'equiv'}_{circ_conf['qubits']}_r{circ_conf['repetition']+prev_rep}"
                dataset_file_path = os.path.join(dataset_folder_path, data_file_name + ".json")
                data_obj = {
                    "name": data_file_name,
                    "data": all_contractions,
                    "date": datetime.today().isoformat(),
                    "version": 2,
                    "experiment_name": experiment_name,
                    "settings": settings,
                    "contraction_settings": contraction_settings,
                    "circuit_settings": circ_conf,
                    "path_settings": path_settings,
                }
                with open(dataset_file_path, "w") as file:
                    json.dump(all_contractions, file, indent=4)          

            if len(sub_tensor_networks) > 1 and settings["simulate"]:
                for sub_tdd in resulting_sub_tdds[1:]:
                    resulting_sub_tdds[0] = cont(resulting_sub_tdds[0], sub_tdd)

                are_equal = tddu.is_tdd_equal(resulting_sub_tdds[0], data["settings"]["state"])
                for data in data_containers:
                    data["equivalence"] = are_equal
                    data["conclusive"] = not are_equal


            data = combinate_data_containers(data_containers)

            if (data["expect_equivalence"] != data["equivalence"]):
                print('\033[31m' + "Erroneous result: Expected != TDD" + '\033[m')
                continue

            if not data["equivalence"] and settings["find_counter"]:
                result_tdd.show(name="final_tdd")
                inds = [v.name for v in result_tdd.index_set]
                trace = tddu.get_counter_example_trace(result_tdd, inds, len(inds)-1)
                counter_state = tddu.convert_trace_to_state_vector(trace)
                simulation_using_counter(circuit, counter_state, deepcopy(data))



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

def simulation_using_counter(circuit, counter_example, data):
    settings = data["settings"]
    # Transform to tensor networks (without initial states and cnot decomp)
    print("Constructing tensor network...")
    tensor_network = tnu.get_tensor_network(circuit, split_cnot=settings["cnot_split"], 
                                            state = counter_example)
    if settings["sliced"]:
        tnu.slice_tensor_network_vertically(tensor_network)

    # Construct the plan from CoTenGra
    print("Find contraction path...")
    path = tnu.get_contraction_path(tensor_network, data)

    #tn_draw.draw_tn(tensor_network, color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'], save_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
    #tnu.draw_contraction_order(tensor_network, path, save_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))

    # Prepare gate TDDs
    print("Preparing gate TDDs...")
    gate_tdds = tddu.get_tdds_from_quimb_tensor_network(tensor_network)

    #tddu.draw_all_tdds(gate_tdds, folder=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
    

    # Contract TDDs + equivalence checking
    print(f"Contracting {len(path)} times...")
    #result_tdd = contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"], save_intermediate_results=True, comprehensive_saving=True, folder_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
    result_tdd = fast_contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"])

    print(f"Simulation finds that the two circuits are: {'equivalent' if data['equivalence'] else 'inequivalent'}")



def combinate_data_containers(containers: list[dict]) -> list[dict]:
    """
    circuit data?
    path_data: {
        original_path: concat
        used_trials: concat
        opt_times: outer concat
        opt_sizes: outer concat
        opt_flops: outer concat
        opt_writes: outer concat
        flops: sum
        size: sum
        path: concat
        dot: give it concat path
    }
    not_same_tensors: concat
    tdd_analysis, correct_example: concat

    path_construction_time: sum
    gate_prep_time: sum
    contraction_time: sum

    equivalence: and
    conclusive and
    sizes: concat

    """
    final_container = deepcopy(containers[0])
    if final_container["path_settings"]["method"] == "cotengra":
        final_container["path_data"]["original_path"] = [item for i in range(len(containers)) for item in containers[i]["path_data"]["original_path"]]
        final_container["path_data"]["used_trials"] = [containers[i]["path_data"]["used_trials"] for i in range(len(containers))]
        final_container["path_data"]["opt_times"] = [containers[i]["path_data"]["opt_times"] for i in range(len(containers))]
        final_container["path_data"]["opt_sizes"] = [containers[i]["path_data"]["opt_sizes"] for i in range(len(containers))]
        final_container["path_data"]["opt_flops"] = [containers[i]["path_data"]["opt_flops"] for i in range(len(containers))]
        final_container["path_data"]["opt_writes"] = [containers[i]["path_data"]["opt_writes"] for i in range(len(containers))]
        final_container["path_data"]["flops"] = sum([containers[i]["path_data"]["flops"] for i in range(len(containers))])
        final_container["path_data"]["size"] = sum([containers[i]["path_data"]["size"] for i in range(len(containers))])
    
    final_container["path"] = [item for i in range(len(containers)) for item in containers[i]["path"]]
    final_container["path_data"]["dot"] = tnu.get_dot_from_path(final_container["path"])

    final_container["not_same_tensors"] = [item for i in range(len(containers)) for item in containers[i]["not_same_tensors"]]
    final_container["tdd_analysis"] = [containers[i]["tdd_analysis"] for i in range(len(containers))]
    final_container["correct_example"] = [containers[i]["correct_example"] for i in range(len(containers))]

    final_container["path_construction_time"] = sum([containers[i]["path_construction_time"] for i in range(len(containers))])
    final_container["gate_prep_time"] = sum([containers[i]["gate_prep_time"] for i in range(len(containers))])
    final_container["contraction_time"] = sum([containers[i]["contraction_time"] for i in range(len(containers))])
    
    final_container["equivalence"] = all([containers[i]["equivalence"] for i in range(len(containers))])
    final_container["conclusive"] = all([containers[i]["conclusive"] for i in range(len(containers))])
    final_container["sizes"] = dict([item for i in range(len(containers)) for item in containers[i]["sizes"].items()])

    if "state" in final_container["settings"]:
        final_container["settings"]["state"] = str(final_container["settings"]["state"])

    return final_container



if __name__ == "__main__":
    contraction_settings = {
                "max_time": 300, # in seconds, -1 for inf
                "max_replans": 1,
                "max_intermediate_node_size": -1 #-1 for inf
            }

    path_settings = {
                "method": "cotengra",
                "opt_method": "all", #  kahypar-balanced, kahypar-agglom, labels, labels-agglom
                "minimize": "flops",
                "max_repeats": 100,
                "max_time": 200,
                "use_proportional": True,
                "gridded": False,
                "linear_fraction": 0,
                "model_name": "experiment_n2"
            }

    iter_settings = {
        "algorithms": ["ghz", "dj", "graphstate"],#["qftentangled", "su2random", "twolocalrandom", "qpeexact", "wstate", "realamprandom"],#,#, "ghz", "graphstate", "qftentangled"],
        "levels": [(0, 2)],
        "qubits": list(range(4,10,1)),#list(range(4,100,1)),#[64, 128, 256],#list(range(256,257,1)),#sorted(list(set([int(x**(3/2)) for x in range(2, 41)])))#list(set([int(2**(x/4)) for x in range(4, 30)]))
        "random_gate_dels_range": [0],
        "repetitions": 1
    }

    settings = {
        "simulate": True,
        "sliced": False,
        "cnot_split": True,
        "use_subnets": True,
        "find_counter": False,
        "use_qcec_only": False
    }

    first_experiment(iter_settings=iter_settings, settings=settings, contraction_settings=contraction_settings, path_settings=path_settings,
                     folder_name="generating_data_set_v2")
