import cotengra as ctg
import random
from quimb.tensor import Circuit
import tn_draw
import random
import os
import numpy as np

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
    circ.apply_gate('X', regs[-1], controls=regs[:-1])

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
    min_id = min(tensor_network.tensor_map.keys())

    for step in reasonable_path:
        i0 = index_map[step[0]]
        i1 = index_map[step[1]]

        usable_path.append((i0, i1))
        index_map[next_index] = i1
        next_index += 1
    
    return usable_path

def contract(tensor_network, path, draw_frequency = -1):
    usable_path = get_usable_path(path, tensor_network)
    for i, step in enumerate(usable_path):
        if draw_frequency > 0 and i % draw_frequency == 0: 
            tensor_network.draw()
        tensor_network._contract_between_tids(step[0], step[1])
    
    return tensor_network.tensor_map[usable_path[-1][1]]

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
    problematic_gates = ["S"]
    for problem_gate in problematic_gates:
        for idx in tensor_network.tag_map[problem_gate]:
            data = tensor_network.tensor_map[idx].data
            data_shape = data.shape
            rectified_data = np.array([rectify_complex(v) for v in data.flatten()]).reshape(data_shape)
            tensor_network.tensor_map[idx]._set_data(rectified_data)

    return tensor_network

def rectify_complex(v: complex, threshold=1e-12) -> complex:
    new_v_real = v.real
    if v.real - round(v.real) < threshold:
        new_v_real = round(v.real)
    
    new_v_imag = v.imag
    if v.imag - round(v.imag) < threshold:
        new_v_imag = round(v.imag)

    return complex(new_v_real, new_v_imag)


def get_contraction_path(tensor_network, data):
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
    
    if path is None:
        raise NotImplementedError(f"Method {settings['method']} is not supported")

    usable_path = get_usable_path(tensor_network, path)
    data["path"] = usable_path

    (verified, msg) = verify_path(usable_path)
    if not verified:
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

    for s in tensor_network.sites:
        i = follow(f"k{s}", -1)
        precurser_depths[i].append(0)
        check_tensor.append(i)

    while check_tensor:
        i = check_tensor.pop(0)
        if i not in tensor_pos and len(precurser_depths[i]) >= int(len(tensor_network.tensor_map[i].shape) / 2):
            rows = [int(tag[1:]) for tag in tensor_network.tensor_map[i].tags if "I" in tag and "PSI" not in tag]
            row = sum(rows) / len(rows)
            tensor_pos[i] = (min(precurser_depths[i]) - width, row)

            for ind in tensor_network.tensor_map[i].inds:
                ii = follow(ind, i)
                if ii not in tensor_pos:
                    precurser_depths[ii].append(tensor_pos[i][0])
                    check_tensor.append(ii)
    
    return tensor_pos 

def draw_contraction_order(tensor_network, usable_path, width = 1.0, save_path = ""):
    ind_contraction_order = get_ind_contraction_order(tensor_network, usable_path)
    tensor_pos = get_tensor_pos(tensor_network, width)

    edge_colors = {}
    node_colors = {}

    min_depth = min([d for d, _ in tensor_pos.values()])

    for tid, pos in tensor_pos.items():
        a = 1 - pos[0] / min_depth
        node_colors[tid] = (a, 1 - a, 4 * a * (1 - a))

    for ind, step in ind_contraction_order.items():
        a = step / (len(usable_path))
        edge_colors[ind] = (a, 1 - a, 4 * a * (1 - a))

    tensor_pos |= {ind : (min_depth - 1 if ind[0] == "b" else 1, int(ind[1:])) for ind in tensor_network.outer_inds()}

    tn_draw.draw_tn(tensor_network, fix = tensor_pos, iterations=0, 
                    node_color=node_colors, edge_colors=edge_colors, edge_scale=5, node_scale=10, save_path = save_path)

def draw_depth_order(tensor_network):
    tensor_depths = get_tensor_pos(tensor_network)

    node_colors = {}

    max_depth = max([d for d, _ in tensor_depths.values()])


    for ind, step in tensor_depths.items():
        a = step[0] / max_depth
        node_colors[ind] = (a, 1 - a, 4 * a * (1 - a))

    tn_draw.draw_tn(tensor_network, iterations=3, initial_layout='kamada_kawai', node_color=node_colors, edge_scale=5, node_scale=10)

if __name__ == "__main__":
    n = 10
    options = [[1 + 0j, 0j], [0j, 1 + 0j]]
    state = [random.choice(options) for _ in range(n)]

    tensor_network = get_tensor_network(get_circuit(n), split_cnot=False, state = state)

    #draw_depth_order(tensor_network)

    usable_path = get_usable_path(tensor_network, tensor_network.contraction_path(ctg.HyperOptimizer(methods = "greedy", minimize="flops", max_repeats=1, max_time=60, progbar=True, parallel=False)))

    draw_contraction_order(tensor_network, usable_path, save_path=os.path.join(os.path.realpath(__file__), "..", "..", "experiments", "plots", "contraction_order"))

    #verified, message = verify_path(usable_path)
    #if not verified:
    #    print("Path Warning: " + message)

    #path = tensor_network.contraction_path(ctg.HyperOptimizer(minimize="flops", max_repeats=128, max_time=60, progbar=True, parallel=False))



