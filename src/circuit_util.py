from mqt.ddsim.pathqasmsimulator import create_tensor_network
from qiskit import QuantumCircuit
from quimb.tensor import Circuit
from quimb.tensor.circuit import Gate
from string import Template
import cotengra as ctg
import numpy as np
import random
from tddpure.TDD.TDD import Ini_TDD, TDD
from tddpure.TDD.TN import Index,Tensor,TensorNetwork
from tddpure.TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import TransformationPass

class UToU3Translator(TransformationPass):

    def run(self, dag):
        for node in dag.op_nodes():
            if node.op.name in ["U", "u"]:
                p1 = node.op.params[0]
                p2 = node.op.params[1]
                p3 = node.op.params[2]

                replacement = QuantumCircuit(1)
                replacement.u3(p1,p2,p3)

                dag.substitute_node_with_dag(node, circuit_to_dag(replacement))

        return dag
    
    def _should_decompose(self, node):
        if node.name in ["U", "u"]:
            return True
        return False

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

def get_qiskit_example_circuit() -> QuantumCircuit:
    circ = QuantumCircuit(3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    return circ

def quimb_to_qiskit_circuit(quimb_circuit: Circuit):
    raise NotImplementedError("This is not supported by quimb")
    circ_qasm = quimb_circuit
    return QuantumCircuit.from_qasm_str(circ_qasm)

def qiskit_to_quimb_circuit(qiskit_circuit: QuantumCircuit):
    circ_qasm = qiskit_circuit.qasm()
    circ_qasm_no_u = circ_qasm.replace("\nu(", "\nu3(")
    return Circuit.from_openqasm2_str(circ_qasm_no_u)

def _quimb_to_qasm(circuit: Circuit):
    temp = Template("")
    
    properties = {

    }

    return temp.substitute(properties)


def get_gate_template(gate: Gate) -> Template:
    temp = Template('$name$params $qubit')
    
    properties = {
        'name' : gate.label,
        'params' : '' if len(gate.params) == 0 else str(gate.params),
        'qubit' : get_qubit_template(gate.qubits)
    }
    return temp.substitute(properties)

def get_qubit_template(qubits):
    if len(qubits) == 1:
        return Template('q[$index]').substitute({'index' : qubits[0]})
    return Template('q[$i1], q[$i2]').substitute({'i1' : qubits[0], 'i2' : qubits[1]})
