from mqt.ddsim.pathqasmsimulator import create_tensor_network
from qiskit import QuantumCircuit
from quimb.tensor import Circuit
import cotengra as ctg
import numpy as np
from tddpure.TDD.TDD import Ini_TDD, TDD
from tddpure.TDD.TN import Index,Tensor,TensorNetwork
from tddpure.TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs

def makeCircuit():
    circ = QuantumCircuit(3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(0, 2)
    return circ

def pureHCircuit():
    circ = QuantumCircuit(1)
    circ.h(0)
    return circ

def pureCNOTCircuit():
    circ = QuantumCircuit(2)
    circ.cx(1, 0)
    return circ

def getContractionPlan():
    opt = ctg.HyperOptimizer(minimize="flops", max_repeats=128, max_time=60, progbar=True, parallel=1)
    circ_qasm = makeCircuit().qasm()
    print(circ_qasm)
    circ = Circuit.from_openqasm2_str(circ_qasm)
    #qft_tensor_network = create_tensor_network(makeCircuit())
    
    network = circ.psi
    network.draw()
    

U = np.array([[1,1],[1,-1]])
var = [Index('x0'),Index('y0')]
ts1 = Tensor(U,var)

Ini_TDD(['x0','y0','x1','y1'])

tdd1 = ts1.tdd()
tdd1.show()

# print("Starting...")
# getContractionPlan()
# print("Done")