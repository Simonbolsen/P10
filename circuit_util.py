from mqt.ddsim.pathqasmsimulator import create_tensor_network
from qiskit import QuantumCircuit
from quimb.tensor import Circuit
import cotengra as ctg
import numpy as np
import random
from tddfork.TDD.TDD import Ini_TDD, TDD
from tddfork.TDD.TN import Index,Tensor,TensorNetwork
from tddfork.TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs

def get_simple_circuit():
    circ = QuantumCircuit(3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(0, 2)
    return circ

def get_example_circuit(n):
    circ = Circuit(n)

    # randomly permute the order of qubits
    regs = list(range(n))
    random.shuffle(regs)

    # hamadard on one of the qubits
    circ.apply_gate('H', regs[0])

    # chain of cnots to generate GHZ-state
    for i in range(n - 1):
        circ.apply_gate('CNOT', regs[i], regs[i + 1])

    # apply multi-controlled NOT
    circ.apply_gate('X', regs[-1], controls=regs[:-1])

    return circ

def quimb_to_qiskit_circuit(quimb_circuit: Circuit):
    circ_qasm = quimb_circuit.qasm()
    return QuantumCircuit.from_qasm_str(circ_qasm)

def qiskit_to_quimb_circuit(qiskit_circuit: QuantumCircuit):
    circ_qasm = qiskit_circuit.qasm()
    return Circuit.from_openqasm2_str(circ_qasm)