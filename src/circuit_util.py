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

all_quimb_gates = ['h', 'x', 'y', 'z', 's', 't', 'cx', 'cnot', 'cy', 'cz', 'rz', 'rx', 'ry', 'sdg', 'tdg', 
                       'x_1_2', 'y_1_2', 'z_1_2', 'w_1_2', 'hz_1_2', 'iden', 'u3', 'u2', 'u1', #'iswap', 'swap', 'cswap', 
                       'cu3', 'cu2', 'cu1', 'fsim', 'fsimg', 'givens', 'rxx', 'ryy', 'rzz', 'crx', 'cry', 'crz',
                       'su4', 'ccx', 'ccnot', 'toffoli', 'ccy', 'ccz', 'fredkin', 'u']

one_qubit_quimb_gates = ['h', 'x', 'y', 'z', 's', 't', 'rz', 'rx', 'ry']
#one_qubit_quimb_gates = ['h', 'x', 'ry']
two_qubit_quimb_gates = ['cx', 'cy', 'cz']
#two_qubit_quimb_gates = ['cx']

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
    circ = QuantumCircuit(2)
    circ.cx(0, 1)
    circ.h(1)
    circ.cx(0, 1)
    circ.cx(0, 1)
    circ.h(0)
    circ.h(1)
    #circ.rz(-1.5707963267948966, 0)
    return circ

def get_other_simple_circuit():
    circ = QuantumCircuit(2)
    circ.cx(0, 1)
    circ.rz(-1.5707963267948966, 1)
    return circ

def get_simple_equiv_circuit():
    circ1 = QuantumCircuit(2)
    circ1.h(0)
    circ1.h(1)
    circ1.cx(0, 1)

    circ2 = QuantumCircuit(2)
    circ2.h(0)
    circ2.h(1)
    circ2.cx(0, 1)
    
    return circ1.compose(circ2.inverse())

def get_example_circuit(n):
    circ = Circuit(n)

    # randomly permute the order of qubits
    regs = list(range(n))
    random.shuffle(regs)

    # hamadard on one of the qubits
    circ.apply_gate('H', regs[0])

    # chain of cnots to generate GHZ-state
    for i in range(n - 1):
        circ.apply_gate('CX', regs[i], regs[i + 1])

    # apply multi-controlled NOT
    circ.apply_gate('X', regs[-1], controls=regs[:-1])

    return circ

def get_qiskit_example_circuit() -> QuantumCircuit:
    circ = QuantumCircuit(3)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    return circ

def handle_gate_str(gate):
    def qubit_str(qubits):
        qubit_list = [f'q[{q}]' for q in list(qubits)]
        qubit_str = ','.join(qubit_list) + ';'
        return qubit_str
    def handle_params(params):
        return f"({','.join([str(i) for i in params])})" if len(params) > 0 else ''
    
    return f"{gate.label.lower().replace('iden', 'id')}{handle_params(gate.params)} {qubit_str(gate.qubits)}"

def get_qasm_header(qubits):
    return f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{qubits}];\n'

def quimb_to_qiskit_circuit(quimb_circuit: Circuit, as_obj: bool = False, gate_prefixes = None):
    out_str = get_qasm_header(quimb_circuit.N)
    gate_strs = [f"{'' if gate_prefixes is None else gate_prefixes[i] + ' '}{handle_gate_str(g)}" for i, g in enumerate(quimb_circuit.gates)]
    res = out_str + '\n'.join(gate_strs)
    return res if not as_obj else QuantumCircuit.from_qasm_str(res)

def qiskit_to_quimb_circuit(qiskit_circuit: QuantumCircuit):
    circ_qasm = qiskit_circuit.qasm()
    circ_qasm_no_u = circ_qasm.replace("\nu(", "\nu3(").replace("\nid ", "\niden ")
    return Circuit.from_openqasm2_str(circ_qasm_no_u)

def qasm_to_quimb_circuit(qasm):
    circ_qasm_no_u = qasm.replace("\nu(", "\nu3(")
    return Circuit.from_openqasm2_str(circ_qasm_no_u)

def refresh_circuit(circuit: Circuit):
    qc = Circuit(circuit.N)
    qc.apply_gates(circuit.gates)
    return qc

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

if __name__ == '__main__':
    qiskit_circ = get_qiskit_example_circuit()
    quimb_circ = qiskit_to_quimb_circuit(qiskit_circ)

    out_str = quimb_to_qiskit_circuit(quimb_circ)
    other_qiskit_circ = QuantumCircuit.from_qasm_str(out_str)

    assert qiskit_circ.qasm() == other_qiskit_circ.qasm()
