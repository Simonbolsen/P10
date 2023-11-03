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
import argparse
from tabulate import tabulate
from bcolors import bcolors, printlc

import matplotlib as mpl
mpl.use("TkAgg")

argparser = argparse.ArgumentParser()

experiment_types = {
    "formal": 0,
    "simulation": 1
}

setup_types = {
    "standard": 0,
    "remove1": 1,
    "remove3": 2,
    "unrelated": 3,
    "mirrored": 4
}

methods = {
    "cotengra": 0,
    "linear": 1,
}

path_methods = {
    "greedy": 0,
    "betweenness": 1,
    "walktrap": 2,
    "linear": 3
}

minimize_methods = {
    "flops": 0,
    "size": 1,
    "combo": 2
}

argparser.add_argument('--exp_name', dest='exp_name', type=str)
argparser.add_argument('--exp_type', dest="exp_type", type=str, default="formal", choices=experiment_types.keys())
argparser.add_argument('-debug', dest="debug", action="store_true")

# --------------- Circuit Settings ---------------
argparser.add_argument('--setup', dest='setup', type=str, default="standard", choices=setup_types.keys())
argparser.add_argument('--algorithms', dest='algorithms', type=[str], default=["dj", "ghz"])
argparser.add_argument('--levels', dest='levels', type=[(int, int)], default=[(0,2)])
argparser.add_argument('--qubits', dest='qubits', type=[(int, int, int)], default=[(2,256,1)], help="Takes tuples of (start, stop, step)")
argparser.add_argument('--gate_deletions', dest='num_of_gate_deletions', type=int, default=0)

# --------------- Contraction Settings ---------------
argparser.add_argument('--max_cont_time', dest='max_cont_time', type=int, default=-1)
argparser.add_argument('--max_attemps', dest='max_attemps', type=int, default=1)
argparser.add_argument('--max_int_node_size', dest='max_int_node_size', type=int, default=-1)

# --------------- Path Settings ---------------
argparser.add_argument('--method', dest='method', type=str, default="cotengra", choices=methods.keys())
argparser.add_argument('--path_method', dest='path_method', type=str, default="greedy", choices=path_methods.keys())
argparser.add_argument('--minimize', dest='minimize', type=str, default='flops', choices=minimize_methods.keys())
argparser.add_argument('--max_repeats', dest='max_repeats', type=int, default=50)
argparser.add_argument('--max_plan_time', dest='max_plan_time', type=int, default=60)
argparser.add_argument('--linear_frac', dest='linear_fraction', type=float, default=0.0)

# --------------- Saving Settings ---------------
argparser.add_argument('-int_res', dest="save_intermediate_results", action="store_true")
argparser.add_argument('-draw', dest="draw", action="store_true")
argparser.add_argument('-print', dest="should_print", action="store_true")


supported_algorithms = [
    "ghz",
    "graphstate",
    #"twolocalrandom",  # No good
    #"qftentangled", # Not working
    "dj",
    #"qpeexact", # Not working
    #"su2random",
    #"wstate",
    #"realamprandom"
]


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
    if not data["simulate"]:
        data["equivalence"] = tddu.is_tdd_identitiy(resulting_tdd)
        data["conclusive"] = True
    else:
        data["equivalence"] = tddu.is_tdd_equal(resulting_tdd, data["state"])
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
        data["equivalence"] = tddu.is_tdd_equal(resulting_tdd, data["state"])
        data["conclusive"] = not data["equivalence"]
    data["sizes"] = sizes

    return resulting_tdd

def get_all_configs(args: argparse.Namespace):
    all_configs = []
    for algorithm in args.algorithms:
        for level in args.levels:
            (start, stop, step) = args.qubits
            for qubit in range(start, stop, step):
                all_configs.append({"algorithm": algorithm, "level": level, "qubits": qubit})

    return all_configs

def prepare_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

def prepare_data_container(args, exp_name, circuit_config):
    options = [[1 + 0j, 0j], [0j, 1 + 0j]]

    data = {
            #"max_rank": circuit_difficulty[circ_conf["algorithm"]] * circ_conf["qubits"],
            "experiment_name": exp_name,
            "experiment_type": args.exp_type,

            "file_name": f"circuit_{circuit_config['algorithm']}_{circuit_config['level'][0]}{circuit_config['level'][1]}_{circuit_config['qubits']}",
            "contraction_settings": {
                "max_time": args.max_cont_time, # in seconds, -1 for inf
                "max_replans": args.max_attempts,
                "max_intermediate_node_size": args.max_int_node_size #-1 for inf
            },
            "circuit_settings": circuit_config,
            "circuit_data": {},
            "state": [random.choice(options) for _ in range(circuit_config["qubits"])] if args.exp_type == "simulation" else None,
            "path_settings": {
                "method": args.method,
                "opt_method": args.path_method, # greedy, betweenness, walktrap
                "minimize": args.minimize,
                "max_repeats": args.max_repeats,
                "max_time": args.max_plan_time,
                "linear_fraction": args.linear_fraction
            },
            "path_data": {},
            "not_same_tensors": [],
            "tdd_analysis": None,
            "correct_example": None
        }

    return data

def cprint(text: str):
    if should_print:
        print(text)

##debug=False
def first_experiment(args: argparse.Namespace):
    # Prepare save folder and file paths
    experiment_name = f"{args.exp_name}_{datetime.today().strftime('%Y-%m-%d_%H-%M')}"
    folder_path = os.path.join("experiments", experiment_name)
    prepare_folder(folder_path)
    print(f"Performing experiment with {args.algorithms} for levels: {args.levels}\n\tqubits: {args.qubits}")

    # Prepare benchmark circuits:
    circuit_configs = get_all_configs(args)

    # For each circuit, run equivalence checking:
    for circ_conf in circuit_configs:
        circ_conf["random_gate_deletions"] = 0
        circ_conf["setup"] = args.setup
        # Prepare data container
        data = prepare_data_container(args, experiment_name, circ_conf)

        working_path = os.path.join(folder_path, data["file_name"])
        prepare_folder(working_path)

        # Prepare circuit
        print("Preparing circuits...")
        starting_time = time.time_ns()
        circuit = bu.get_dual_circuit_setup_quimb(data, draw=False)
        data["circuit_setup_time"] = int((time.time_ns() - starting_time) / 1000000)

        # Transform to tensor networks (without initial states and cnot decomp)
        print("Constructing tensor network...")
        starting_time = time.time_ns()
        tensor_network = tnu.get_tensor_network(circuit, split_cnot=False, state = data["state"])
        data["tn_construnction_time"] = int((time.time_ns() - starting_time) / 1000000)
        
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

            # Contract TDDs + equivalence checking
            print(f"Contracting {len(path)} times...")
            starting_time = time.time_ns()
            #result_tdd = contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"], save_intermediate_results=False, comprehensive_saving=False, folder_path=os.path.join(working_path, data["file_name"] + f"_R{attempts}"))
            result_tdd = fast_contract_tdds(gate_tdds, data, max_time=data["contraction_settings"]["max_time"])
            data["contraction_time"] = int((time.time_ns() - starting_time) / 1000000)

            if (data["qcec_equivalence"] != data["equivalence"]):
                print('\033[31m' + "Erroneous result: QCEC != TDD" + '\033[m')

            # Save data for circuit
            if not args.debug:
                print("Saving data...")
                file_path = os.path.join(working_path, data["file_name"] + f"_R{attempts}" + ".json")
                with open(file_path, "w") as file:
                    json.dump(data, file, indent=4)

            #result_tdd.show(name=os.path.join(working_path, data["file_name"] + f"_R{attempts}" + "_TDD"))
            if result_tdd is not None:
                succeeded = True
            else:
                print(f"Retry #{attempts+1}")

def legal_args(args):
    for algorithm in args.algorithms:
        if not algorithm in supported_algorithms:
            print(f"Algorithm {algorithm} not supported.")
            return False

    for (left, right) in args.levels:
        if not (left in range(4) and right in range(4)):
            print(f"Level ({left}, {right}) is not valid.")
            return False

    for (start, stop, step) in args.qubits:
        if start < 2 or start >= stop:
            print(f"Qubits must be <=2 and stop must be greater than start")
            return False
        
    if args.num_of_gate_deletions < 0:
        print(f"Gate deletions cannot be negative")
        return False
    
    if args.linear_fraction < 0.0 or args.linear_fraction > 1.0:
        print(f"Linear fraction must be in [0.0, 1.0]")
        return False
    
    


    return True

def pretty_print(args):
    return tabulate(vars(args).items(), headers=["arg", "value"], missingval=f"{bcolors.WARNING}None{bcolors.ENDC}")


def run_experiment(args):
    if not legal_args(args):
        raise argparse.ArgumentError("Illegal arguments")

    pretty_print(args)

    global should_print
    should_print = args.should_print



if __name__ == "__main__":
    args = argparser.parse_args()
    run_experiment(args)
    #first_experiment()
