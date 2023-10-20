import cotengra as ctg
import random
from quimb.tensor import Circuit
import tn_draw

def get_circuit(n):
    circ = Circuit(n)

    # randomly permute the order of qubits
    regs = list(range(n))
    random.shuffle(regs)

    # hamadard on one of the qubits
    circ.apply_gate('H', regs[0])

    # chain of cnots to generate GHZ-state
    for i in range(n - 1):
        circ.apply_gate('CNOT', regs[i], regs[i + 1])

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

def get_tensor_network(circuit, include_state = True, split_cnot = True):
    
    if include_state:
        tensor_network = circuit.psi
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

    #tensor_network.tensor_map = {i: t for i, t in enumerate(tensor_network.tensor_map.values())}

    return tensor_network

def get_contraction_path(tensor_network, data):
    path = None
    settings = data["path_settings"]
    if settings["method"] == "cotengra":
        optimiser = ctg.HyperOptimizer(methods=settings["opt_method"], minimize=settings["minimize"], max_repeats=settings["max_repeats"], 
                               max_time=settings["max_time"], progbar=True, parallel=False)
        tree = tensor_network.contraction_tree(optimiser)
        path = tree.get_path() 

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


if __name__ == "__main__":
    tensor_network = get_tensor_network(get_circuit(10), include_state = True, split_cnot=False)

    tn_draw.draw_tn(tensor_network, iterations=0, initial_layout='kamada_kawai', 
                        highlight_inds=[i for i in tensor_network.outer_inds() + tensor_network.inner_inds()])

    #tensor_network.draw(iterations=0, initial_layout='kamada_kawai', 
    #                    highlight_inds=[i for i in tensor_network.outer_inds() + tensor_network.inner_inds()])

    #usable_path = get_usable_path(tensor_network, tensor_network.contraction_path(ctg.HyperOptimizer(minimize="flops", max_repeats=128, max_time=60, progbar=True, parallel=False)))

    #verified, message = verify_path(usable_path)
    #if not verified:
    #    print("Path Warning: " + message)

    #path = tensor_network.contraction_path(ctg.HyperOptimizer(minimize="flops", max_repeats=128, max_time=60, progbar=True, parallel=False))



