from quimb.tensor import Circuit
import cotengra as ctg
import numpy as np
import random
from tddfork.TDD.TDD import Ini_TDD, TDD
from tddfork.TDD.TN import Index,Tensor,TensorNetwork
from tddfork.TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs
import circuit_util as cu
import tdd_util as tddu
import tensor_network_util as tnu

def contract_tdds(tdds, usable_path):

    sizes = {i: [0, tdd.node_number()] for i, tdd in tdds.items()}

    for left_index, right_index in usable_path:
        tdds[right_index] = tddu.cont(tdds[left_index], tdds[right_index])
        sizes[right_index].append(tdds[right_index].node_number())

    resulting_tdd = tdds[right_index]


if __name__ == "__main__":
    n = 3

    

    def get_tdd(n):
        circ = Circuit(n)

        # randomly permute the order of qubits
        regs = list(range(n))
        random.shuffle(regs)

        # hamadard on one of the qubits
        circ.apply_gate('H', regs[0])

        tn = circ.get_uni()
        tnm = tn.tensor_map
        tensor = tnm[list(tnm.keys())[0]]

        Ini_TDD(list(tn.all_inds()))
        t = Tensor(tensor.data, [Index(s) for s in tensor.inds])
        return t.tdd()

    def get_identity_tdd(n):
        shape = [2] * (2 * n)
        identity_tensor = np.eye(np.prod(shape[n:])).reshape(shape)

        str_inds = [str(s) for s in range(len(shape))]
        inds = [Index(s) for s in str_inds]

        Ini_TDD(str_inds)
        t = Tensor(identity_tensor, inds)
        return t.tdd()

    tdd1 = get_tdd(n)
    tdd2 = get_identity_tdd(n)

    circ = cu.get_qiskit_example_circuit()
    quimb_circ = cu.qiskit_to_quimb_circuit(circ)

    tn = quimb_circ.psi
    tdd = tddu.circ_to_tdd(circ)
    tdd.show()

    # print("Starting...")
    # getContractionPlan()
    # print("Done")

