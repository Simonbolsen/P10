import cotengra as ctg
import random
from quimb.tensor import Circuit, drawing
import tn_draw
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
from graph_util import to_nx_graph, tag_tn

def get_circuit(n):
    circ = Circuit(n)

    # randomly permute the order of qubits
    regs = list(range(n))

    # chain of cnots to generate GHZ-state
    for d in range(2 * n):
        for i in range(n - 1):
            if random.random() > 0.3:
                circ.apply_gate('H', regs[i])
            circ.apply_gate('CNOT', regs[i], regs[i + 1])
            if random.random() > 0.3:
                circ.apply_gate('H', regs[i])

    # apply multi-controlled NOT
    #circ.apply_gate('X', regs[-1], controls=regs[:-1])

    return circ

def get_nontriv_identity_circuit(n):
    circ = Circuit(n)

    # randomly permute the order of qubits
    regs = list(range(n))

    # chain of cnots to generate GHZ-state
    for d in range(n - 1):
        circ.apply_gate('H', regs[d])
        circ.apply_gate('CNOT', regs[d], regs[d + 1])

    circ.apply_gate('H', regs[0])

    for d in range(n - 2, -1, -1):
        circ.apply_gate('CNOT', regs[d], regs[d + 1])
        circ.apply_gate('H', regs[d])

    return circ
    
def get_subgraph_containing_circuit(n):
    circ = Circuit(n)

    # randomly permute the order of qubits
    regs = list(range(n))

    middle = int(n / 2)
    # chain of cnots to generate GHZ-state
    for d in range(middle):
        circ.apply_gate('H', regs[d])
        circ.apply_gate('CNOT', regs[d], regs[d + 1])

    for d in range(middle+1, n-1):
        circ.apply_gate('H', regs[d])
        circ.apply_gate('CNOT', regs[d], regs[d + 1])

    circ.apply_gate('H', regs[0])

    for d in range(middle-1, -1, -1):
        circ.apply_gate('CNOT', regs[d], regs[d + 1])
        circ.apply_gate('H', regs[d])

    for d in range(n-2, middle+1, -1):
        circ.apply_gate('CNOT', regs[d], regs[d + 1])
        circ.apply_gate('H', regs[d])

    return circ

def get_reasonable_path(path):
    num_of_tensors = len(path) + 1
    reasonable_path = []
    index_map = [i for i in range(num_of_tensors)]

    for step in path:
        reasonable_path.append((index_map[step[0]], index_map[step[1]]))
        index_map.append(num_of_tensors)
        num_of_tensors += 1
        index_map.pop(step[1])
        index_map.pop(step[0])
    
    return reasonable_path

def get_usable_path(tensor_network, path):
    reasonable_path = get_reasonable_path(path)
    usable_path = []
    index_map = {i: t for i, t in enumerate(tensor_network.tensor_map.keys())}
    next_index = len(path) + 1

    for step in reasonable_path:
        i0 = index_map[step[0]]
        i1 = index_map[step[1]]

        usable_path.append((i0, i1))
        index_map[next_index] = i1
        next_index += 1
    
    return usable_path

def contract(tensor_network, path=None, usable_path=None, draw_frequency = -1):
    if usable_path is None:
        if path is None:
            raise Exception("Path must be given to tensor network contraction")
        usable_path = get_usable_path(path, tensor_network)
    for i, step in enumerate(usable_path):
        if draw_frequency > 0 and i % draw_frequency == 0: 
            tensor_network.draw()
        tensor_network._contract_between_tids(step[0], step[1])
    
    return tensor_network.tensor_map[usable_path[-1][1]]

def slice_tensor_network_vertically(tensor_network):
    fill_identity(tensor_network)
    grid = get_grid(tensor_network)

    grid = np.array(grid).transpose().tolist()

    for column in grid:
        for c, cell in enumerate(column):
            if c < len(column) - 1:
                tensor_network._contract_between_tids(cell, column[c+1])

def fill_identity(tensor_network):
    grid = get_grid(tensor_network)
    row_from_tid = get_row_dict(tensor_network)

    def replace(inds, old, new):
        return tuple([new if i == old else i for i in inds])
    
    def is_pair(ind):
        ts = list(tensor_network.ind_map[ind])
        return len(ts) > 1 and row_from_tid[ts[0]] != row_from_tid[ts[1]]

    identity = [[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]]

    new_ind_count = 0

    for r, row in enumerate(grid):
        out_ind = f"k{r}"
        ind = out_ind
        ind_pos = -1
        moved_ind = ind
        for c, cell in enumerate(row):
            if cell == -1:
                new_ind = f"grid_ind_{new_ind_count}"
                new_tag = f"grid_tag_{new_ind_count}"
                t = Tensor(identity, inds=(ind, new_ind), tags=(new_tag, f"I{r}"))
                old_ind = ind
                ind = new_ind
                if old_ind == out_ind:
                    tensor_network.ind_map.pop(out_ind)
                else:
                    tensor_network.ind_map[old_ind] = oset([ind_pos]) if ind_pos != -1 else oset([])
                tensor_network.ind_map[new_ind] = oset()
                tensor_network.add_tensor(t, virtual = True)
                tid = list(tensor_network._get_tids_from_tags([new_tag]))[0]
                row_from_tid[tid] = r
                new_ind_count += 1

                
                ind_pos = tid

                if c == len(row) - 1:
                    tensor_network.ind_map[ind] = oset([tid])
            else:
                t = tensor_network.tensor_map[cell]
                if ind not in t.inds:
                    t._inds = replace(t.inds, moved_ind, ind)
                    tensor_network.ind_map[ind] = oset([cell, ind_pos])
                ind_pos = cell
                stuck = True
                for i in t.inds:
                    if i != ind and not is_pair(i):
                        moved_ind = i
                        ind = i
                        stuck = False
                        break
                if stuck and c < len(row) - 1:
                    raise Exception("No usable ind")
                    
def get_grid_pos(tensor_network, width = 1.0):
    grid_pos = {}

    grid = get_grid(tensor_network)

    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            grid_pos[cell] = (-x * width , -y)

    return grid_pos

def get_row_dict(tensor_network):
    return {k:[int(tag[1:]) for tag in t.tags if "I" in tag and "PSI" not in tag][0] for k,t in tensor_network.tensor_map.items()}

def get_grid(tensor_network): 
    grid = []
    in_grid = {}
    check_tensor = []
    
    def follow(ind, r):
        l = list(tensor_network.ind_map[ind])
        for i in l:
            if i not in in_grid and row_from_tid[i] == r:
                check_tensor.append(i)
        return l
    
    def follow_all(i):
        for ind in tensor_network.tensor_map[i].inds:
            follow(ind, row_from_tid[i])

    tensor_pair_tags = {}
    tensor_pair_pos = {}

    row_from_tid = get_row_dict(tensor_network)

    def set_pair_pos(i, x_pos):
        tag = tensor_pair_tags[i]
        if tensor_pair_pos[tag] is None:
            tensor_pair_pos[tag] = x_pos
        else: 
            l = list(tensor_network.tag_map[tag])
            r0 = row_from_tid[l[0]]
            r1 = row_from_tid[l[1]]

            column = max(tensor_pair_pos[tag], x_pos)

            while len(grid[r0]) < column:
                grid[r0].append(-1)

            while len(grid[r1]) < column:
                grid[r1].append(-1)

            grid[r0].append(l[0])
            grid[r1].append(l[1])
            in_grid[l[0]] = True
            in_grid[l[1]] = True
            follow_all(l[0])
            follow_all(l[1])

    for tag, tensor_set in tensor_network.tag_map.items():
        if "GATE" in tag and len(tensor_set) == 2:
            l = list(tensor_set)
            tensor_pair_tags[l[0]] = tag
            tensor_pair_tags[l[1]] = tag
            tensor_pair_pos[tag] = None

    for s in tensor_network.sites:
        i = list(tensor_network.ind_map[f"k{s}"])[0]
        if i in tensor_pair_tags:
            grid.append([-1])
            set_pair_pos(i, 0)
        else:
            grid.append([i])
            in_grid[i] = True
            follow_all(i)
        
    while check_tensor:
        i = check_tensor.pop(0)
        if i not in in_grid:
            r = row_from_tid[i]
            if i in tensor_pair_tags:
                set_pair_pos(i, len(grid[r]))
            else:
                grid[r].append(i)
                in_grid[i] = True
                follow_all(i)

    max_len = max([len(row) for row in grid])

    for row in grid:
        while len(row) < max_len:
            if type(tensor_network) == TensorNetworkGenVector:
                row.insert(-1, -1)
            else:
                row.append(-1)

    return grid

def get_tensor_network(circuit, split_cnot = True, state = None):
    if state is not None:
        tensor_network = circuit.psi

        for i, q in enumerate(state):
            tensor_network.tensor_map[i]._set_data(q)
    else:
        tensor_network = circuit.get_uni(transposed = True)
        

    if not split_cnot:
        cnot_tag = "CX"
        pairs = []
        for i, tensor_1 in tensor_network.tensor_map.items():
            if cnot_tag in tensor_1.tags:
                done = False
                for ii, tensor_2 in tensor_network.tensor_map.items():
                    if cnot_tag in tensor_2.tags:
                        for tag in tensor_1.tags:
                            if "GATE" in tag and tag in tensor_2.tags:
                                pairs.append((i, ii))
                                done = True
                                break
                    if done:
                        break
        for pair in pairs:
            tensor_network._contract_between_tids(pair[0], pair[1])

        cz_tag = "CZ"
        pairs = []
        for i, tensor_1 in tensor_network.tensor_map.items():
            if cz_tag in tensor_1.tags:
                done = False
                for ii, tensor_2 in tensor_network.tensor_map.items():
                    if cz_tag in tensor_2.tags:
                        for tag in tensor_1.tags:
                            if "GATE" in tag and tag in tensor_2.tags:
                                pairs.append((i, ii))
                                done = True
                                break
                    if done:
                        break
        for pair in pairs:
            tensor_network._contract_between_tids(pair[0], pair[1])

    # Fixing problematic gates (S, ...?)
    problematic_gates = []#["S"]
    for problem_gate in problematic_gates:
        if problem_gate in tensor_network.tag_map:
            for idx in tensor_network.tag_map[problem_gate]:
                data = tensor_network.tensor_map[idx].data
                data_shape = data.shape
                rectified_data = np.array([rectify_complex(v) for v in data.flatten()]).reshape(data_shape)
                tensor_network.tensor_map[idx]._set_data(rectified_data)

    return tensor_network

def rectify_complex(v: complex, threshold=1e-12) -> complex:
    new_v_real = v.real
    if abs(v.real - round(v.real)) < threshold:
        new_v_real = round(v.real)
    
    new_v_imag = v.imag
    if abs(v.imag - round(v.imag)) < threshold:
        new_v_imag = round(v.imag)

    return complex(new_v_real, new_v_imag)

def get_linear_path(tensor_network, fraction = 0.0, gridded = False):
    tensor_pos = get_grid_pos(tensor_network) if gridded else get_tensor_pos(tensor_network)

    pairs = []

    for ts in tensor_network.ind_map.values():
        ts = list(ts)
        if len(ts) > 1:
            pairs.append((*ts, (tensor_pos[ts[0]][0] + tensor_pos[ts[1]][0]) / 2))

    pairs = sorted(pairs, key=lambda x:x[2])

    part_of = {i:i for i in tensor_network.tensor_map.keys()}

    def get_current_tensor(i):
        p = part_of[i]
        return p if p == i else get_current_tensor(p)

    path = []

    while len(pairs) > 0:
        pair = pairs.pop(int(len(pairs) * fraction))
        left = get_current_tensor(pair[0])
        right = get_current_tensor(pair[1])

        if left != right:
            path.append((left, right))
            part_of[left] = right

    return path

def get_random_path(tensor_network, gridded = False):
    tensor_pos = get_grid_pos(tensor_network) if gridded else get_tensor_pos(tensor_network)

    pairs = []

    for ts in tensor_network.ind_map.values():
        ts = list(ts)
        if len(ts) > 1:
            pairs.append((*ts,))

    random.shuffle(pairs)

    part_of = {i:i for i in tensor_network.tensor_map.keys()}

    def get_current_tensor(i):
        p = part_of[i]
        return p if p == i else get_current_tensor(p)

    path = []

    while len(pairs) > 0:
        pair = pairs.pop(0)
        left = get_current_tensor(pair[0])
        right = get_current_tensor(pair[1])

        if left != right:
            path.append((left, right))
            part_of[left] = right

    return path

def get_nn_path(tn: TensorNetwork, circuit, data):
    settings = data["path_settings"]
    model_path = os.path.join("models", settings["model_name"] + ".pt")
    if (not os.path.isfile(model_path)):
        print(f"Could not find model: {model_path}")

    model = gnn.load_model(model_path)
    
    first_circ_gate_count = data["circuit_data"]["unrolled_first_circ_gate_count"]
    tag_tn(tn, circuit, first_circ_gate_count)

    graph = to_nx_graph(tn)
    torch_graph = from_networkx(graph)

    pred_path = gnn.get_path(model, torch_graph)

    return pred_path

def get_contraction_path(tensor_network, circuit, data):
    path = None
    settings = data["path_settings"]
    if settings["method"] == "cotengra":
        optimiser = ctg.HyperOptimizer(methods=settings["opt_method"], minimize=settings["minimize"], max_repeats=settings["max_repeats"], 
                               max_time=settings["max_time"], progbar=True, parallel=False)
        tree = tensor_network.contraction_tree(optimiser)
        path = tree.get_path() 

        data["path_data"]["original_path"] = path
        data["path_data"]["used_trials"] = len(optimiser.times)
        data["path_data"]["opt_times"] = optimiser.times
        data["path_data"]["opt_sizes"] = optimiser.costs_size
        data["path_data"]["opt_flops"] = optimiser.costs_flops
        data["path_data"]["opt_writes"] = optimiser.costs_write

        data["path_data"]["flops"] = tree._flops
        data["path_data"]["size"] = tree._sizes._max_element
    
        usable_path = get_usable_path(tensor_network, path)
        
    elif settings["method"] == "linear":
        print(f"Linear fraction: {settings['linear_fraction']}")
        usable_path = get_linear_path(tensor_network, settings["linear_fraction"] if "linear_fraction" in settings else 0.0, gridded=settings["gridded"])
    elif settings["method"] == "nn_model":
        print(f"Using NN-model {settings['model_name']}. Loading")
        usable_path = get_nn_path(tensor_network, circuit, data)
    elif settings["method"] == "random":
        usable_path = get_random_path(tensor_network, gridded=settings["gridded"])
    else:
        raise NotImplementedError(f"Method {settings['method']} is not supported")

    data["path"] = usable_path
    (verified, msg) = verify_path(usable_path)
    if not verified:
        print(usable_path)
        raise AssertionError(f"Not valid path: {msg}")

    return usable_path

def test(tensor_network, path):
    s = contract(tensor_network.copy(deep = True), path)
    flat_s = s.data.flatten()

    t = tensor_network.contract(optimize = path)
    flat_t = t.transpose(*s.inds, inplace = True).data.flatten()

    assert (flat_s == flat_t).all(), f"{flat_s}\n{flat_t}"
    print("Test Succesful!")

def verify_path(usable_path):
    usable_path = usable_path.copy()
    usable_path.reverse()
    indeces = [usable_path[0][1]]
    
    for step in usable_path:
        if step[1] not in indeces:
            return False, f"Index {step[1]} is not included in the final tensor"
        if step[0] in indeces:
            return False, f"Index {step[0]} is contracted more than once"
        indeces.append(step[0])

    return True, ""

def get_dot_from_path(input_list, wrong_nodes=[]):
    # Initialize an empty list to store the transformed strings
    transformed_list = []

    # Iterate through the sublists and format them as required
    for sublist in input_list:
        if len(sublist) == 2:
            transformed_list.append(f'a{sublist[0]} -> a{sublist[1]}')

    wrong_nodes_str = "\n".join(["a" + str(wrong_node) + "[style = filled, color = red]" for wrong_node in wrong_nodes])

    # Join the transformed strings with semicolons and return the result
    return "https://dreampuf.github.io/GraphvizOnline/#" + urllib.parse.quote("digraph G {" + '; '.join(transformed_list) + ";\n\n" + wrong_nodes_str + "}")

def get_ind_contraction_order(tensor_network, usable_path):
    inds_by_tensor_index = {i : t.inds  for i, t in tensor_network.tensor_map.items()}
    ind_contraction_order = {}


    for i, step in enumerate(usable_path):
        contracted_inds = list(set(inds_by_tensor_index[step[0]]) & set(inds_by_tensor_index[step[1]]))
        inds_by_tensor_index[step[1]] += inds_by_tensor_index[step[0]]

        for ind in contracted_inds:
            ind_contraction_order[ind] = i

    for ind in tensor_network.outer_inds():
        ind_contraction_order[ind] = len(usable_path)

    return ind_contraction_order

def get_tensor_pos(tensor_network, width = 1.0):
    precurser_depths = {i : [] for i in tensor_network.tensor_map.keys()}
    tensor_pos = {}
    check_tensor = []
    
    def follow(ind, i):
        values = list(tensor_network.ind_map[ind])
        return values[0] if len(values) == 1 else (values[0] if values[0] != i else values[1])

    sites_present = tensor_network.site_inds_present if type(tensor_network) == TensorNetworkGenVector else tensor_network.upper_inds_present
    for s in sites_present:
        i = follow(s, -1)
        precurser_depths[i].append(0)
        check_tensor.append(i)

    rows = {k:[int(tag[1:]) for tag in t.tags if "I" in tag and "PSI" not in tag] for k,t in tensor_network.tensor_map.items()}

    while check_tensor:
        i = check_tensor.pop(0)
        if i not in tensor_pos and len(precurser_depths[i]) >= int(len(tensor_network.tensor_map[i].shape) / 2):
            
            row = sum(rows[i]) / len(rows[i])
            tensor_pos[i] = (min(precurser_depths[i]) - width, -row)

            for ind in tensor_network.tensor_map[i].inds:
                ii = follow(ind, i)
                if ii not in tensor_pos and len(set(rows[i]) & set(rows[ii])) > 0:
                    precurser_depths[ii].append(tensor_pos[i][0])
                    check_tensor.append(ii)
    
    return tensor_pos 

def draw_contraction_order(tensor_network, usable_path, width = 1.0, save_path = ""):
    ind_contraction_order = get_ind_contraction_order(tensor_network, usable_path)
    tensor_pos = get_grid_pos(tensor_network, width)

    edge_colors = {}
    node_colors = {}

    min_depth = min([d for d, _ in tensor_pos.values()])

    for tid, pos in tensor_pos.items():
        a = 1 - pos[0] / min_depth
        node_colors[tid] = (1,1,1)#(a, 1 - a, 4 * a * (1 - a))

    for ind, step in ind_contraction_order.items():
        a = step / (len(usable_path))
        b = 1-a
        edge_colors[ind] = (0.29 * a + 0.1 * b, b, 0.56 * a + 0.1*b)

    for k in tensor_network.outer_inds():
        edge_colors[k] = (0.2,0.2,0.2)

    tensor_pos |= {ind : (min_depth - 1 if ind[0] == "b" else 1, -int(ind[1:])) for ind in tensor_network.outer_inds()}

    tn_draw.draw_tn(tensor_network, fix = tensor_pos, iterations=0, 
                    node_color=node_colors, edge_colors=edge_colors, edge_scale=5, node_scale=1.5, margin=0.5, save_path = save_path, arrow_length = 0.0)

def draw_depth_order(tensor_network):
    tensor_depths = get_tensor_pos(tensor_network)

    node_colors = {}

    max_depth = max([d for d, _ in tensor_depths.values()])


    for ind, step in tensor_depths.items():
        a = step[0] / max_depth
        node_colors[ind] = (a, 1 - a, 4 * a * (1 - a))

    tn_draw.draw_tn(tensor_network, iterations=3, initial_layout='kamada_kawai', node_color=node_colors, edge_scale=5, node_scale=10)

def plot_igraph(G: ig.Graph):
    fig, ax = plt.subplots()
    ig.plot(G, target=ax)
    plt.show()

def plot_components(components):
    fig, ax = plt.subplots()
    ig.plot(
        components,
        target=ax,
        palette=ig.RainbowPalette(),
        vertex_size=7,
        vertex_color=list(map(int, ig.rescale(components.membership, (0, 200), clamp=True))),
        edge_width=0.7
    )
    plt.show()

def find_and_split_subgraphs_in_tn(tn: TensorNetwork, draw=False) -> ig.Graph:
    edges = [tuple(e) for e in tn.ind_map.values() if len(e) > 1]
    G = ig.Graph(edges)
    components = G.connected_components(mode="weak")
    if draw:
        plot_components(components)

    used_tags = []
    for (i, tensor) in tn.tensor_map.items():
        new_tag = f"Subgraph_{components.membership[i]}"
        tensor.add_tag(new_tag)
        used_tags.append(new_tag)
    
    used_tags = list(set(used_tags))

    sub_tns = []
    for tag in used_tags:
        sub_tns.append(tn.select(tag, which="any", virtual=False))

    if draw:
        for sub_tn in sub_tns:
            sub_tn.draw()

    return sub_tns

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
    # circuit = bu.get_dual_circuit_setup_quimb(data, draw=False)
    # #circuit = get_circuit(5)

    # tensor_network = get_tensor_network(circuit, split_cnot=True, state = None)
    # tensor_network.draw()

    # add_multigate_order_tags(tensor_network, circuit)

    # G = to_nx_graph(tensor_network, True)

    # rgreedy = get_usable_path(tensor_network, tensor_network.contraction_path(
    #     ctg.HyperOptimizer(methods = "random-greedy", minimize="flops", max_repeats=1, max_time=60, progbar=False, parallel=False)))
        
    # add_usable_path_to_graph(G, tensor_network, rgreedy, "rgreedy")
    
    # tensor_network.draw()
    # #tensor_network = get_tensor_network(get_subgraph_containing_circuit(n), split_cnot=False, state = None)
    # #tensor_network.draw()
    # #find_and_split_subgraphs_in_tn(tensor_network)
    # #draw_depth_order(tensor_network)

    # naive = get_linear_path(tensor_network, fraction=0.0, gridded=True)
    # prop = get_linear_path(tensor_network, fraction=0.5, gridded=True)
    # #print(verify_path(usable_path))

    # rgreedy = get_usable_path(tensor_network, tensor_network.contraction_path(
    #     ctg.HyperOptimizer(methods = "rgreedy", minimize="flops", max_repeats=1, max_time=60, progbar=False, parallel=False)))
    # betweennes = get_usable_path(tensor_network, tensor_network.contraction_path(
    #     ctg.HyperOptimizer(methods = "betweenness", minimize="flops", max_repeats=1, max_time=60, progbar=False, parallel=False)))

    # draw_contraction_order(tensor_network, naive, width = 0.5) #save_path=os.path.join(os.path.realpath(__file__), "..", "..", "experiments", "plots", "contraction_order"))
    # draw_contraction_order(tensor_network, prop, width = 0.5)
    # draw_contraction_order(tensor_network, rgreedy, width = 0.5)
    # draw_contraction_order(tensor_network, betweennes, width = 0.5)

    # # slice_tensor_network_vertically(tensor_network)
    # # usable_path = get_linear_path(tensor_network, fraction=0.8)
    # # draw_contraction_order(tensor_network, usable_path, width = 0.5)

    # # result = contract(tensor_network, usable_path = usable_path)

    # # print(result)

    # #verified, message = verify_path(usable_path)
    # #if not verified:
    # #    print("Path Warning: " + message)

    # #path = tensor_network.contraction_path(ctg.HyperOptimizer(minimize="flops", max_repeats=128, max_time=60, progbar=True, parallel=False))



