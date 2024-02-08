import bench_util as bu
import tensor_network_util as tnu
import graph_util as gu
import cotengra as ctg
import os
from tqdm import tqdm

algorithms = [
    #"ghz",
    #"graphstate",
    "twolocalrandom",  
    "qftentangled", 
    #"dj",
    "qpeexact", 
    "su2random",
    #"wstate",
    "realamprandom"
]

def generate_mutation_set(algorithms: [str], qubit_range: (int, int, int), method: str = "betweenness"):
    folder_path = os.path.join("dataset", "graphs")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    mutation_degree = lambda x : (1 + x) * 0.015
    for algo in tqdm(algorithms, position=1, leave=False):
        for qubit in tqdm(range(*qubit_range), position=0, leave=False):
            for mut_idx in range(10):
                data, circ = bu.get_combined_circuit_example(algo, qubit)
                circ = bu.mutate_circuit(circ, mutation_degree(mut_idx), data)

                first_circ_gate_count = data["circuit_data"]["unrolled_first_circ_gate_count"]
                tensor_network = tnu.get_tensor_network(circ, split_cnot=True, state = None)
                
                gu.tag_tn(tensor_network, circ, first_circ_gate_count)

                G = gu.to_nx_graph(tensor_network, draw=False)
                
                if method == "betweenness":
                    sub_tensor_networks = tnu.find_and_split_subgraphs_in_tn(tensor_network)
                    data["sub_networks"] = len(sub_tensor_networks)

                    for stn in sub_tensor_networks:
                        betweennes = tnu.get_usable_path(stn, stn.contraction_path(
                            ctg.HyperOptimizer(methods = "betweenness", minimize="flops", max_repeats=1, max_time=60, progbar=False, parallel=False)))  
                        gu.add_usable_path_to_graph(G, stn, betweennes, "betweenness")
                
                name = f"graph_{algo}_q{qubit}_mdi{mut_idx}"
                gu.save_nx_graph(G, folder_path, name)


if __name__ == "__main__":
    generate_mutation_set(algorithms, (5, 26, 1))

