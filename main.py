from mqt.ddsim.pathqasmsimulator import create_tensor_network
from qiskit import QuantumCircuit
from quimb.tensor import Circuit
import cotengra as ctg
 
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
    

print("Starting...")
getContractionPlan()
print("Done")