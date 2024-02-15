import cotengra as ctg
import random
from quimb.tensor import Circuit, drawing
import tn_draw
import random
import os
import numpy as np
import urllib.parse
import math
from quimb.tensor.tensor_arbgeom import TensorNetworkGenVector
from quimb.tensor import Tensor, TensorNetwork, oset
import igraph as ig
import matplotlib.pyplot as plt
import bench_util as bu
import networkx as nx
from tqdm import tqdm
import graph_nn as gnn
from torch_geometric.utils.convert import from_networkx
import tensor_network_util as tnu
import re

def add_multigate_order_tags(tn: TensorNetwork, circuit: Circuit):
    gates = {f"GATE_{i}":[f"I{q}" for q in list(gate.qubits)] for i, gate in enumerate(circuit.gates) if len(gate.qubits) > 1}

    for gate_id, gate in gates.items():
        for i, qubit_index in enumerate(gate):
            gate_tag_ind = tn.tag_map[gate_id]
            qubit_tag_ind = tn.tag_map[qubit_index]
            tensor_id = list(gate_tag_ind & qubit_tag_ind)[0]
            tensor = tn.tensor_map[tensor_id]
            gate_type_tag = f"{extract_gate_tag_from_tn_tags(tensor.tags)}_{i}"
            tensor.add_tag(gate_type_tag)

def draw_nx_graph(G: nx.Graph):
    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    ax0.set_title("Plot with NetworkX draw")
    nx.draw_kamada_kawai(G, node_size=50, ax=ax0)
    plt.axis("off")
    plt.show()


def extract_gate_tag_from_tn_tags(tags: oset([str]), specific_tag: bool = False):
    tag_list = list(tags)
    filtered_tags = [tag for tag in tag_list if tag.lower() in bu.all_quimb_gates]

    if specific_tag:
        all_gate_tags = [tag for tag in tag_list if filtered_tags[0] in tag]
        if min(all_gate_tags) in max(all_gate_tags) and any(char.isdigit() for char in max(all_gate_tags)):
            filtered_tags = [max(all_gate_tags)]

    assert len(filtered_tags) == 1
    return filtered_tags[0]

def is_split_from_tn_tags(tags):
    return int("Split" in tags)

def is_first_from_tn_tags(tags):
    return int("First" in tags)

def is_last_from_tn_tags(tags):
    return int("Last" in tags)

def is_last_circuit_from_tn_tags(tags):
    return int("Circ2" in tags)

def to_nx_graph(tn: TensorNetwork, draw=False):
    edges = [tuple(e) for e in tn.ind_map.values() if len(e) > 1]
    graph = nx.Graph(edges)

    node_tags_dict = {node_id: extract_gate_tag_from_tn_tags(tn.tensor_map[node_id].tags, specific_tag=True) for node_id in graph.nodes}
    split_tags_dict = {node_id: is_split_from_tn_tags(tn.tensor_map[node_id].tags) for node_id in graph.nodes}
    first_tags_dict = {node_id: is_first_from_tn_tags(tn.tensor_map[node_id].tags) for node_id in graph.nodes}
    last_tags_dict = {node_id: is_last_from_tn_tags(tn.tensor_map[node_id].tags) for node_id in graph.nodes}
    circuit_tags_dict = {node_id: is_last_circuit_from_tn_tags(tn.tensor_map[node_id].tags) for node_id in graph.nodes}
    shape_dict = {node_id: tn.tensor_map[node_id].shape for node_id in graph.nodes}
    
    nx.set_node_attributes(graph, node_tags_dict, "gate")
    nx.set_node_attributes(graph, split_tags_dict, "is_in_split")
    nx.set_node_attributes(graph, first_tags_dict, "is_first_tensor")
    nx.set_node_attributes(graph, last_tags_dict, "is_last_tensor")
    nx.set_node_attributes(graph, circuit_tags_dict, "is_last_circuit")
    nx.set_node_attributes(graph, shape_dict, "shape")

    if draw:
        draw_nx_graph(graph)
    
    return graph

def add_usable_path_to_graph(graph: nx.Graph, tn: TensorNetwork, usable_path, method_name: str):
    ind_path = tnu.get_ind_contraction_order(tn, usable_path)
    tn_edges = {tuple(nodes): ind_path[tag] for tag, nodes in tn.ind_map.items() if len(list(nodes)) == 2}

    nx.set_edge_attributes(graph, tn_edges, method_name)


def add_circuit_tag_to_tn(tn: TensorNetwork, half: int):
    current_index = 0
    true_half = half
    tn_map_keys = list(tn.tensor_map.keys())
    while current_index < true_half:
        actual_index = tn_map_keys[current_index:current_index+1][0]
        true_half += 0.5 * (len(tn.tensor_map[actual_index].shape) - 2)
        current_index += 1
    
    #true_half = [0.5 * len(tn.tensor_map[tid].shape) for tid in list(tn.tensor_map.keys())[:half]]
    true_half = int(true_half)
    for i in list(tn.tensor_map.keys())[:true_half]:
        tn.tensor_map[i].add_tag("Circ1")
    for i in list(tn.tensor_map.keys())[true_half:]:
        tn.tensor_map[i].add_tag("Circ2")

def add_split_tag_to_tn(tn: TensorNetwork, qubits):
    circ_1_tensors = list(tn._get_tids_from_tags("Circ1", "any"))[::-1]
    circ_2_tensors = list(tn._get_tids_from_tags("Circ2", "any"))

    last_in_circ_1 = find_first_tensor_for_each_qubit(tn, circ_1_tensors, qubits)
    first_in_circ_2 = find_first_tensor_for_each_qubit(tn, circ_2_tensors, qubits)

    split_tensors = last_in_circ_1 + first_in_circ_2

    for tensor in split_tensors:
        tn.tensor_map[tensor].add_tag("Split") 

def add_edge_tag_to_tn(tn: TensorNetwork):
    first_tensors = []
    last_tensors = []

    for tid, tensor in tn.tensor_map.items():
        if any([re.match(r"k\d+", ind) for ind in list(tensor.inds)]):
            tensor.add_tag("First")
            first_tensors.append(tid)
        elif any([re.match(r"k\d+", ind) for ind in list(tensor.inds)]):
            tensor.add_tag("Last")
            last_tensors.append(tid)

    return (first_tensors, last_tensors)

def find_first_tensor_for_each_qubit(tn: TensorNetwork, tids: [int], qubits):
    first_tensors = []
    free_qubits = list(range(qubits))
    for tid in tids:
        tensor_qubits = [int(tag[1:]) for tag in tn.tensor_map[tid].tags if re.match(r"I\d+", tag)]
        if any([tensor_qubit in free_qubits for tensor_qubit in tensor_qubits]):
            first_tensors.append(tid)
            free_qubits = [fq for fq in free_qubits if fq not in tensor_qubits]

    return first_tensors

def tag_tn(tn: TensorNetwork, circ: Circuit, first_circ_gate_count: int):
    add_circuit_tag_to_tn(tn, first_circ_gate_count)

    add_split_tag_to_tn(tn, circ.N)
    add_edge_tag_to_tn(tn)

    add_multigate_order_tags(tn, circ)

def circuit_to_nx_graph(circuit: Circuit, data):
    pass

def generate_graph_files(algorithms, qubits, folder_path="graphs"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    settings = {
        "simulate": False,
        "level": (0, 2),
        "random_gate_deletions": 0
    }

    for algorithm in algorithms:
        settings["algorithm"] = algorithm
        for qubit in tqdm(qubits):
            settings["qubits"] = qubit
            data = {
                "circuit_settings": settings,
                "path_settings": {
                    "use_proportional": True
                },
                "circuit_data": {}
            }

            name = f"graph_{settings['algorithm']}_q{settings['qubits']}"

            circuit = bu.get_dual_circuit_setup_quimb(data, draw=False)
            first_circ_gate_count = data["circuit_data"]["unrolled_first_circ_gate_count"]
            tensor_network = tnu.get_tensor_network(circuit, split_cnot=True, state = None)
            
            tag_tn(tensor_network, circuit, first_circ_gate_count)

            G = to_nx_graph(tensor_network, draw=False)

            # rgreedy = get_usable_path(tensor_network, tensor_network.contraction_path(
            #     ctg.HyperOptimizer(methods = "random-greedy", minimize="flops", max_repeats=1000000, max_time=60, progbar=False, parallel=False)))

            sub_tensor_networks = tnu.find_and_split_subgraphs_in_tn(tensor_network)
            data["sub_networks"] = len(sub_tensor_networks)

            for stn in sub_tensor_networks:
                betweennes = tnu.get_usable_path(stn, stn.contraction_path(
                    ctg.HyperOptimizer(methods = "betweenness", minimize="flops", max_repeats=1, max_time=600, progbar=False, parallel=False)))  
                add_usable_path_to_graph(G, stn, betweennes, "betweenness")

            save_nx_graph(G, folder_path, name)


def save_nx_graph(graph: nx.Graph, path: str, file_name: str):
    final_path = file_name + ".gml" if path == "" else os.path.join(path, file_name + ".gml")
    nx.write_gml(graph, final_path)

def load_nx_graph(path: str):
    graph = nx.read_gml(path)
    return graph

algorithms = [
    "ghz",
    "graphstate",
    #"twolocalrandom",  
    #"qftentangled", 
    "dj",
    #"qpeexact", 
    #"su2random",
    "wstate",
    #"realamprandom"
]

if __name__ == "__main__":
    
    settings = {
        "simulate": False,
        "algorithm": "ghz",
        "level": (0, 2),
        "qubits": 5,
        "random_gate_deletions": 0
    }
    data = {
        "circuit_settings": settings,
        "path_settings": {
            "use_proportional": True
        },
        "circuit_data": {

        }
    }
    name = f"graph_{settings['algorithm']}_q{settings['qubits']}"

    generate_graph_files(algorithms, list(range(5,51,5)), os.path.join("graphs", "debug"))