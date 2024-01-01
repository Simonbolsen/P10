from qiskit import QuantumCircuit
from tddpure.TDD.TDD import Ini_TDD, cont
from tddpure.TDD.TN import Index,Tensor,TensorNetwork
from tddpure.TDD.TDD_Q import cir_2_tn
import os
import numpy as np
import tddpure.TDD.ComplexTable as ct
import tddpure.TDD.TDD as TDD
from quimb.tensor import Tensor as QTensor
from tqdm import tqdm
from copy import deepcopy

def tn_to_tdd(tn: TensorNetwork):
    return tn.cont()

def circ_to_tdd(circuit: QuantumCircuit):
    tn, indices = cir_2_tn(circuit)
    Ini_TDD(index_order=indices)
    return tn.cont()

def tdd_contract(tdd1: TDD, tdd2: TDD):
    return cont(tdd1, tdd2)

def reverse_lexicographic_key(s):
        return (len(s), s[1:], s[0])

def get_tdds_from_quimb_tensor_network(tensor_network) -> dict[int,TDD.TDD]:
    tdds = {}

    for i, tensor in tqdm(tensor_network.tensor_map.items()):
        #tensor_t = tensor.transpose(*(sorted(list(tensor.inds), key=reverse_lexicographic_key, reverse=False)))
        t = Tensor(tensor.data, [Index(s) for s in tensor.inds])
        tdds[i] = t.tdd()
        #check = tensor_of_tdd(tdds[i])
        #same = np.allclose(check.data, tensor.transpose(*check.inds).data)

    return tdds


def draw_all_tdds(tdds: dict[int,TDD.TDD], folder="tdds_images"):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    for key, tdd in tdds.items():
        file_path = os.path.join(folder, "tdd_" + str(key))
        tdd.show(name=file_path)

def get_identity_tdd(inds):
    shape = [2] * len(inds)
    n = int(len(inds) / 2)
    identity_tensor = np.transpose(np.eye(np.prod(shape[n:])).reshape(shape), 
                                    [int(i / 2) + (i % 2) * n for i in range(n * 2)])
    t = Tensor(identity_tensor, inds)
    return t.tdd() if n <= len(TDD.global_index_order) else None

def to_complex(z):
    return z.r.val + z.i.val * 1j

def tensor_of_tdd(tdd):    
    ki_set = tdd.key_2_index
    idx_names = [ki_set[i] for i in range(len(ki_set))]#[idx.name for idx in tdd.index_set]
    t = tensor_of_node(tdd.node, to_complex(tdd.weight), idx_names, ki_set)
    return QTensor(t, idx_names[::-1])#.transpose(*sorted(list(), key=reverse_lexicographic_key, reverse=True))

def tensor_of_node(node, weight, inds, key_index_set):
    if node == TDD.terminal_node:
        if inds == []:
            return weight
        else:
            t = tensor_of_node(node, weight, inds[:-1], key_index_set)
            return [t, t]

    ind = inds[-1]

    if key_index_set[node.key] == ind:
        left = tensor_of_node(node.succ[0], weight * to_complex(node.out_weight[0]), inds[:-1], key_index_set)
        right = tensor_of_node(node.succ[1], weight * to_complex(node.out_weight[1]), inds[:-1], key_index_set)
    else:
        left = tensor_of_node(node, weight, inds[:-1], key_index_set)
        right = left

    return [left, right]

def is_tdd_equal(tdd, tensor):
     return is_node_equal(tdd.node, tensor)

def is_node_equal(node, tensor):
    if node == TDD.terminal_node:
        return True
    t = tensor[0]

    if node.out_weight[0] == TDD.cn1 and t[0] == 1:
        return is_node_equal(node.succ[0], tensor[1:])
    
    if node.out_weight[1] == TDD.cn1 and t[1] == 1:
        return is_node_equal(node.succ[1], tensor[1:])

    return False

def is_tdd_identitiy(node, expected_length = -1):
    if type(node) == TDD.TDD:
        return is_node_identitiy(node.node, expected_length == -1, expected_length)
    if node == TDD.terminal_node:
        return expected_length == 0 or expected_length is None
    return False

def is_node_identitiy(node, length_indifferent, expected_length):
    if node == TDD.terminal_node or (not length_indifferent and expected_length < 0):
        return length_indifferent or expected_length == 0
    left_node = node.succ[0]
    right_node = node.succ[1]

    return (node.out_weight[0] == ct.cn1 and 
            node.out_weight[1] == ct.cn1 and
            left_node.out_weight is not None and right_node.out_weight is not None and
            left_node.out_weight[0] == ct.cn1 and
            left_node.out_weight[1] == ct.cn0 and
            right_node.out_weight[0] == ct.cn0 and
            right_node.out_weight[1] == ct.cn1 and
            left_node.succ[0] == right_node.succ[1] and
            is_node_identitiy(left_node.succ[0], length_indifferent, -1 if length_indifferent else expected_length - 1))


def get_counter_example_trace(node, inds, expected_length):
    if type(node) == TDD.TDD:
        incomplete_trace = get_counter_example_trace_rec(node.node, inds, expected_length, [], False)
        if incomplete_trace is None or incomplete_trace == []:
            return []
        trace_counter = 0
        complete_trace = []
        for i in range(incomplete_trace[0][0],-1,-1):
            if (trace_counter < len(incomplete_trace) and incomplete_trace[trace_counter][0] == i):
                complete_trace.append(incomplete_trace[trace_counter])
                trace_counter += 1
            else:
                complete_trace.append((i, 0, inds[i]))
        return complete_trace
    return []

def get_counter_example_trace_rec(node, inds, expected_length, current_trace, found_counter):
    if node == TDD.terminal_node or expected_length < 0:
        return current_trace if found_counter else []
    left_node = node.succ[0]
    right_node = node.succ[1]

    partial_trace = []
    next_node = None

    if left_node.out_weight is None or right_node.out_weight is None:
        print("Invalid TDD. Cannot find trace")
        return None
        

    if node.out_weight[0] != ct.cn1:
        partial_trace.append((node.key, 0, inds[node.key]))
        next_node = left_node
        if left_node.out_weight[0] != ct.cn0:
            partial_trace.append((left_node.key, 0, inds[left_node.key]))
            next_node = left_node.succ[0]
        elif left_node.out_weight[1] != ct.cn0:
            partial_trace.append((left_node.key, 1, inds[left_node.key]))
            next_node = left_node.succ[1]
        else:
            partial_trace = []
            next_node = None
    elif node.out_weight[1] != ct.cn1:
        partial_trace.append((node.key, 1, inds[node.key]))
        next_node = right_node
        if right_node.out_weight[0] != ct.cn0:
            partial_trace.append((right_node.key, 0, inds[right_node.key]))
            next_node = right_node.succ[0]
        elif right_node.out_weight[1] != ct.cn0:
            partial_trace.append((right_node.key, 1, inds[right_node.key]))
            next_node = right_node.succ[1]
        else:
            partial_trace = []
            next_node = None
    
    if len(partial_trace) == 0:
        if left_node.out_weight[0] != ct.cn1:
            partial_trace.append((node.key, 0, inds[node.key]))
            partial_trace.append((left_node.key, 0, inds[left_node.key]))
            next_node = left_node.succ[0]
        elif left_node.out_weight[1] != ct.cn0:
            partial_trace.append((node.key, 0, inds[node.key]))
            partial_trace.append((left_node.key, 1, inds[left_node.key]))
            next_node = left_node.succ[1]
        elif right_node.out_weight[0] != ct.cn0:
            partial_trace.append((node.key, 1, inds[node.key]))
            partial_trace.append((right_node.key, 0, inds[right_node.key]))
            next_node = right_node.succ[0]
        elif right_node.out_weight[1] != ct.cn1:
            partial_trace.append((node.key, 1, inds[node.key]))
            partial_trace.append((right_node.key, 1, inds[right_node.key]))
            next_node = right_node.succ[1]
    
    new_found_counter = found_counter or len(partial_trace) > 0
    if len(partial_trace) == 0:
        next_node = left_node.succ[0]
        partial_trace.append((node.key, 0, inds[node.key]))
        partial_trace.append((left_node.key, 0, inds[left_node.key]))

    return get_counter_example_trace_rec(next_node, inds, expected_length - len(partial_trace), current_trace + partial_trace, new_found_counter)

def convert_trace_to_state_vector(trace):
    indices = [t[1] for t in trace if "b" in t[2]]
    state_vector = [[1 + 0j, 0j] if i == 0 else [0j, 1 + 0j] for i in indices]
    return state_vector

def convert_trace_to_state_tensor(trace, inds):
    state_vector = convert_trace_to_state_vector(trace)
    state_tensor = QTensor(state_vector, inds[::-1][0::2])
    state_tensor._set_data(state_tensor.data.transpose())
    return state_tensor

if __name__ == "__main__":
    inds = list(np.array([[("k" if o == 0 else "b") + str(i) for o in range(2)] for i in range(2)]).flatten())
    sorted_inds = sorted(inds, key=reverse_lexicographic_key, reverse=True)
    Ini_TDD(sorted_inds)

    I = get_identity_tdd([Index(i) for i in sorted_inds])
    I.node.succ[0].out_weight[0] = ct.non_cn1_cn0
    I.show(name="counter_example_test")

    m = tensor_of_tdd(I)

    print(is_tdd_identitiy(I))
    counter_trace = get_counter_example_trace(I, sorted_inds, len(sorted_inds)-1)
    print(counter_trace)

    state_tensor = convert_trace_to_state_tensor(counter_trace, sorted_inds)

    res = m @ state_tensor
    assert (res.data != state_tensor.data).any()
    print("?")