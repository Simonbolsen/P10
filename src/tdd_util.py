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

def tn_to_tdd(tn: TensorNetwork):
    return tn.cont()

def circ_to_tdd(circuit: QuantumCircuit):
    tn, indices = cir_2_tn(circuit)
    Ini_TDD(index_order=indices)
    return tn.cont()

def tdd_contract(tdd1: TDD, tdd2: TDD):
    return cont(tdd1, tdd2)

def reverse_lexicographic_key(s):
        return (len(s), s[::-1])

def get_tdds_from_quimb_tensor_network(tensor_network) -> dict[int,TDD.TDD]:
    variable_order = sorted(list(tensor_network.all_inds()), key=reverse_lexicographic_key, reverse=True)
    Ini_TDD(variable_order, max_rank=len(variable_order)+1)
    print(f"Using rank {len(variable_order)+1} for TDDs")
    
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


def is_tdd_identitiy(node):
    if type(node) == TDD.TDD:
        node = node.node
    if node == TDD.terminal_node:
        return True
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
            is_tdd_identitiy(left_node.succ[0]))

if __name__ == "__main__":
    inds = [str(i) for i in range(4)]
    Ini_TDD(inds)

    I = get_identity_tdd([Index(i) for i in inds])

    m = tensor_of_tdd(I)

    print("?")