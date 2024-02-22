import bench_util as bu
import tensor_network_util as tnu
import graph_util as gu
import cotengra as ctg
import os
from tqdm import tqdm
import random as rnd
from tddpure.TDD.TDD import Ini_TDD, TDD, tdd_2_np, cont
from tdd_util import reverse_lexicographic_key
from tddpure.TDD.TN import Index,Tensor,TensorNetwork

algorithms = [
    "ghz",
    #"graphstate",
    #"twolocalrandom",  
    #"qftentangled", 
    #"dj",
    #"qpeexact", 
    #"su2random",
    #"wstate",
    #"realamprandom"
]

def is_any_node_split(G):
    for id, node in G._node.items():
        if node["gate"] == "Split":
            return True
    return False


def generate_mutation_set(algorithms: [str], qubit_range: (int, int, int), method: str = "betweenness"):
    folder_path = os.path.join("dataset", "dump")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    mutation_degree = lambda x : (1 + x) * 0.015
    for algo in tqdm(algorithms, position=1, leave=False):
        for qubit in tqdm(range(*qubit_range), position=0, leave=False):
            for mut_idx in range(10):
                for _ in range(1):
                    data, circ = bu.get_combined_circuit_example(algo, qubit)
                    circ = bu.mutate_circuit(circ, mutation_degree(mut_idx), data)

                    first_circ_gate_count = data["circuit_data"]["unrolled_first_circ_gate_count"]
                    tensor_network = tnu.get_tensor_network(circ, split_cnot=True, state = None)
                    
                    gu.tag_tn(tensor_network, circ, first_circ_gate_count)

                    G = gu.to_nx_graph(tensor_network, draw=False)
                    
                    if is_any_node_split(G):
                        print("h")

                    if method == "betweenness":
                        sub_tensor_networks = tnu.find_and_split_subgraphs_in_tn(tensor_network)
                        data["sub_networks"] = len(sub_tensor_networks)

                        for stn in sub_tensor_networks:
                            betweennes = tnu.get_usable_path(stn, stn.contraction_path(
                                ctg.HyperOptimizer(methods = "betweenness", minimize="flops", max_repeats=1, max_time=60, progbar=False, parallel=False)))  
                            gu.add_usable_path_to_graph(G, stn, betweennes, "betweenness")
                    
                    name = f"graph_{algo}_q{qubit}_mdi{mut_idx}"
                    gu.save_nx_graph(G, folder_path, name)



def generate_random_tensor(indices, qubits=5, num_of_gates=100):
    circ = bu.get_random_circuit(qubits, num_of_gates)
    tensor_network = tnu.get_tensor_network(circ, split_cnot=True, state = None)

    data = {
        "settings": {},
        "contraction_settings": {},
        "circuit_settings": {},
        "circuit_data": {},
        "path_settings": {},
        "path_data": {},
        "not_same_tensors": [],
    }

    data["path_settings"] = {
                "method": "cotengra",
                "opt_method": "random-greedy", #  kahypar-balanced, kahypar-agglom, labels, labels-agglom
                "minimize": "flops",
                "max_repeats": 1,
                "max_time": 600,
                "use_proportional": True,
                "gridded": False,
                "linear_fraction": 0,
                "model_name": "experiment_n2"
            }

    path = tnu.get_contraction_path(tensor_network, circ, data)

    quimb_result = tensor_network.contract(optimize=data["path_data"]["original_path"])
    
    variable_order = sorted(list(quimb_result.inds), key=reverse_lexicographic_key, reverse=True)
    processed_result = quimb_result.transpose(*variable_order, inplace=False)
    quimb_result_tdd = Tensor(processed_result.data, [Index(s) for s in processed_result.inds]).tdd()
    return None

def generate_random_tdd(indices, shared_indices, max_qubits):
    # Sample qubits but with bias towards lower values
    qubit = max_qubits

    # Sample num of gates relative to the number of qubits (normal dist with variance)
    num_of_gates = 100

    # Create indices the TDD should use (including the shared indices)
    used_indices = shared_indices + rnd.sample(list(set(indices) - set(shared_indices)), 2 * qubit - len(shared_indices))
    used_indices.sort()

    return generate_random_tensor(used_indices, qubits=qubit, num_of_gates=num_of_gates).tdd()


def generate_contraction_data(datapoints=10000, max_qubits=100, max_shared_indices=3):
    global_indices = [f'a{i:09d}' for i in range(2*(max_qubits+1))]

    for dp in range(datapoints):
        shared_indices = [global_indices[i] for i in rnd.sample(range(max_qubits+1), rnd.randint(1, max_shared_indices))]
        left_tdd, left_stats = generate_random_tdd(global_indices, shared_indices, max_qubits)
        right_tdd, right_stats = generate_random_tdd(global_indices, shared_indices, max_qubits)

        resulting_tdd = cont(left_tdd, right_tdd)
        result_size = resulting_tdd.node_number()


if __name__ == "__main__":
    #generate_mutation_set(algorithms, (5, 6, 1))
    generate_contraction_data()
