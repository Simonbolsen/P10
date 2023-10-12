import cotengra as ctg
import random
from quimb.tensor import Circuit

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

def get_usable_path(path, tensor_network):
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

def test(tensor_network, path):
    s = contract(tensor_network.copy(deep = True), path)
    flat_s = s.data.flatten()

    t = tensor_network.contract(optimize = path)
    flat_t = t.transpose(*s.inds, inplace = True).data.flatten()

    assert (flat_s == flat_t).all(), f"{flat_s}\n{flat_t}"
    print("Test Succesful!")

tensor_network = get_tensor_network(get_circuit(10), include_state = False, split_cnot=False)

tree = tensor_network.contraction_tree(ctg.HyperOptimizer(minimize="flops", max_repeats=128, max_time=60, progbar=True, parallel=False))

path = tree.get_path()
reasonable_path = get_reasonable_path(path)
usable_path = get_usable_path(path, tensor_network)

test(tensor_network, path)

#path = tensor_network.contraction_path(ctg.HyperOptimizer(minimize="flops", max_repeats=128, max_time=60, progbar=True, parallel=False))



