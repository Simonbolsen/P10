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




tn, indices = cir_2_tn(cu.get_example_circuit(5))

U = np.array([[1,1],[1,-1],[-1,-1]])
var = [Index('x0'),Index('y0')]
ts1 = Tensor(U,var)

Ini_TDD(index_order=indices)
tdd = tn.cont()
tdd.show()

# print("Starting...")
# getContractionPlan()
# print("Done")