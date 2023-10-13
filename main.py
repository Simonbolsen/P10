from mqt.ddsim.pathqasmsimulator import create_tensor_network
from qiskit import QuantumCircuit
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

if __name__ == "__main__":
    circ = cu.get_qiskit_example_circuit()
    quimb_circ = cu.qiskit_to_quimb_circuit(circ)

    tn = quimb_circ.psi
    tdd = tddu.circ_to_tdd(circ)
    tdd.show()

    # print("Starting...")
    # getContractionPlan()
    # print("Done")

