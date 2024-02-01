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
            data["failed"] = True
            data["conclusive"] = False
            print("Time limit for contraction reached. Aborting check")
            return None
        
        tdds[right_index] = cont(tdds[left_index], tdds[right_index])

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

def get_all_configs(settings):
    all_configs = []
    for algorithm in settings["algorithms"]:
        for level in settings["levels"]:
            for qubit in settings["qubits"]:
                for dels in settings["random_gate_dels_range"]:
                    for i in range(settings["repetitions"]):
                        all_configs.append({"algorithm": algorithm, "level": level, "qubits": qubit, "random_gate_deletions": dels, "repetition": i})

    return all_configs


def run_qcec(data):
    print("Starting QCEC sanity check")
    starting_time = time.time_ns()
    data["equivalence"] = verify(data["circuit_data"]["circuit_1_qasm"], data["circuit_data"]["circuit_2_qasm"]).equivalence.value in [1,4,5]  # see https://mqt.readthedocs.io/projects/qcec/en/latest/library/EquivalenceCriterion.html
    data["qcec_time"] = int((time.time_ns() - starting_time) / 1000000)
    print(f"QCEC says: {data['equivalence']}")

    data["circuit_data"]["circuit_1_qasm"] = data["circuit_data"]["circuit_1_qasm"].qasm()
    data["circuit_data"]["circuit_2_qasm"] = data["circuit_data"]["circuit_2_qasm"].qasm()

    print("Saving data...")
    file_path = os.path.join(data["working_path"], data["file_name"] + ".json")
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
        


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
            "failed": False
        }

        if "simulate" in settings and settings["simulate"]:
            options = [[1 + 0j, 0j], [0j, 1 + 0j]]
            settings["state"] = [random.choice(options) for _ in range(circ_conf["qubits"])]
        
        data["working_path"] = os.path.join(folder_path, data["file_name"])
        if not os.path.exists(data["working_path"]):
            os.makedirs(data["working_path"], exist_ok=True)

        # Prepare circuit
        print("Preparing circuits...")
        #circuit = bu.get_circuit_setup_quimb(bu.get_benchmark_circuit(circ_conf), draw=False)
        starting_time = time.time_ns()
        circuit = bu.get_dual_circuit_setup_quimb(data, draw=False)
        data["circuit_setup_time"] = int((time.time_ns() - starting_time) / 1000000)

        if data["settings"]["use_qcec_only"]:
            run_qcec(data)
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

            data["circuit_data"]["circuit_1_qasm"] = data["circuit_data"]["circuit_1_qasm"].qasm()
            data["circuit_data"]["circuit_2_qasm"] = data["circuit_data"]["circuit_2_qasm"].qasm()


            resulting_sub_tdds = []
            data_containers = [deepcopy(data) for _ in sub_tensor_networks]
            for i, stn in enumerate(sub_tensor_networks):
                data = data_containers[i]

                # Construct the plan from CoTenGra
                print("Find contraction path...")
                starting_time = time.time_ns()
                path = tnu.get_contraction_path(stn, data)
                data["path_construction_time"] = int((time.time_ns() - starting_time) / 1000000)

                # Prepare gate TDDs
                print("Preparing gate TDDs...")
                starting_time = time.time_ns()
                gate_tdds = tddu.get_tdds_from_quimb_tensor_network(stn)
                data["gate_prep_time"] = int((time.time_ns() - starting_time) / 1000000)

                # Contract TDDs + equivalence checking
                print(f"Contracting {len(path)} times...")
                starting_time = time.time_ns()
                result_tdd = fast_contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"])
                data["contraction_time"] = int((time.time_ns() - starting_time) / 1000000)

                data_containers[i] = data

                if len(sub_tensor_networks) > 1:
                    resulting_sub_tdds.append(result_tdd)

            if any([data_containers[i]["failed"] for i in range(len(data_containers))]):
                continue


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
                file_path = os.path.join(data["working_path"], data["file_name"] + f"_R{attempts}" + ".json")
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
                "opt_method": "greedy", #  kahypar-balanced, kahypar-agglom, labels, labels-agglom
                "minimize": "flops",
                "max_repeats": 1000000000,
                "max_time": 600,
                "use_proportional": True,
                "gridded": False,
                "linear_fraction": 0
            }

    iter_settings = {
        "algorithms": ["qftentangled", "qpeexact"],#, "dj", "graphstate"],#, "ghz", "graphstate", "qftentangled"],
        "levels": [(0, 2)],
        "qubits": [8],#list(range(4,100,1)),#[64, 128, 256],#list(range(256,257,1)),#sorted(list(set([int(x**(3/2)) for x in range(2, 41)])))#list(set([int(2**(x/4)) for x in range(4, 30)]))
        "random_gate_dels_range": [0],
        "repetitions": 1
    }

    settings = {
        "simulate": False,
        "sliced": False,
        "cnot_split": False,
        "use_subnets": True,
        "find_counter": False,
        "use_qcec_only": False
    }

    first_experiment(iter_settings=iter_settings, settings=settings, contraction_settings=contraction_settings, path_settings=path_settings,
                     folder_name="retesting_old_heuristics")
