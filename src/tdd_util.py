from qiskit import QuantumCircuit
from tddpure.TDD.TDD import Ini_TDD, cont
from tddpure.TDD.TN import Index,Tensor,TensorNetwork
from tddpure.TDD.TDD_Q import cir_2_tn
import os
import numpy as np
import tddpure.TDD.ComplexTable as ct
import tddpure.TDD.TDD as TDD
from quimb.tensor import Tensor as QTensor

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
    Ini_TDD(variable_order, max_rank=5000)
    
    tdds = {}

    for i, tensor in tensor_network.tensor_map.items():
        tensor_t = tensor.transpose(*(sorted(list(tensor.inds), key=reverse_lexicographic_key, reverse=True)))
        t = Tensor(tensor_t.data, [Index(s) for s in tensor_t.inds])
        tdds[i] = t.tdd()
        check = tensor_of_tdd(tdds[i])
        same = check.inds == tensor_t.inds
        same = same and np.allclose(check.data, tensor_t.data)
        ...

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
    t = tensor_of_node(tdd.node, to_complex(tdd.weight), tdd.index_set[::-1], tdd.index_set)
    idx_names = [idx.name for idx in tdd.index_set]
    return QTensor(t, idx_names)#.transpose(*sorted(list(), key=reverse_lexicographic_key, reverse=True))

def tensor_of_node(node, weight, inds, index_set):
    if node == TDD.terminal_node:
        if inds == []:
            return weight
        else:
            t = tensor_of_node(node, weight, inds[1:], index_set)
            return [t, t]

    ind = inds[0]

    if index_set[node.key].name == ind.name:
        left = tensor_of_node(node.succ[0], weight * to_complex(node.out_weight[0]), inds[1:], index_set)
        right = tensor_of_node(node.succ[1], weight * to_complex(node.out_weight[1]), inds[1:], index_set)
    else:
        left = tensor_of_node(node, weight, inds[1:], index_set)
        right = left

    return [left, right]

def is_tdd_equal(tdd, tensor):
     return False #TODO

def is_tdd_identitiy(node):
        if type(node) == TDD.TDD:
            node = node.node
        if node == TDD.terminal_node:
            return True
        left_node = node.succ[0]
        right_node = node.succ[1]

        return (node.out_weight[0] == ct.cn1 and 
                node.out_weight[1] == ct.cn1 and
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