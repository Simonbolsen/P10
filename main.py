from quimb.tensor import Circuit
import cotengra as ctg
import numpy as np
import random
from tddfork.TDD.TDD import Ini_TDD
from tddfork.TDD.TN import Index,Tensor,TensorNetwork
from tddfork.TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs
import circuit_util as cu
import tdd_util as tddu
import tensor_network_util as tnu

def contract_tdds(tdds, data):
    usable_path = data["path"]
    sizes = {i: [0, tdd.node_number()] for i, tdd in tdds.items()}

    for left_index, right_index in usable_path:
        tdds[right_index] = tddu.cont(tdds[left_index], tdds[right_index])
        sizes[right_index].append(tdds[right_index].node_number())

    resulting_tdd = tdds[right_index]
    data["equivalence"] = tddu.is_tdd_identitiy(resulting_tdd)
    data["sizes"] = sizes

if __name__ == "__main__":
     
    circ = cu.get_qiskit_example_circuit()
    quimb_circ = cu.qiskit_to_quimb_circuit(circ)


