from mqt.ddsim.pathqasmsimulator import create_tensor_network
from qiskit import QuantumCircuit
from tddfork.TDD.TDD import Ini_TDD, cont
from tddfork.TDD.TN import Index,Tensor,TensorNetwork
from tddfork.TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs
import circuit_util as cu
import os
import numpy as np
import tddfork.TDD.ComplexTable as ct
import tddfork.TDD.TDD as TDD

def tn_to_tdd(tn: TensorNetwork):
    return tn.cont()

def circ_to_tdd(circuit: QuantumCircuit):
    tn, indices = cir_2_tn(circuit)
    Ini_TDD(index_order=indices)
    return tn.cont()

def tdd_contract(tdd1: TDD, tdd2: TDD):
    return cont(tdd1, tdd2)

def reverse_lexicographic_key(s):
        return s[::-1]

def get_tdds_from_quimb_tensor_network(tensor_network):
    variable_order = sorted(list(tensor_network.all_inds()), key=reverse_lexicographic_key)
    Ini_TDD(variable_order)
    
    tdds = {}

    for i, tensor in tensor_network.tensor_map.items():
        t = Tensor(tensor.data, [Index(s) for s in tensor.inds])
        tdds[i] = t.tdd()

    return tdds


def draw_all_tdds(tdds: dict[int,TDD.TDD]):
    folder_name = "tdds_images"
    for key, tdd in tdds.items():
        file_path = os.path.join(folder_name, "tdd_" + str(key))
        tdd.show(name=file_path)

def get_identity_tdd(inds):
    shape = [2] * len(inds)
    n = int(len(inds) / 2)
    identity_tensor = np.transpose(np.eye(np.prod(shape[n:])).reshape(shape), 
                                    [int(i / 2) + (i % 2) * n for i in range(n * 2)])
    t = Tensor(identity_tensor, inds)
    return t.tdd() if n <= len(TDD.global_index_order) else None

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