from mqt.bench import CompilerSettings, QiskitSettings, TKETSettings
from mqt.bench import get_benchmark
from quimb.tensor import Circuit
import tdd_util as tddu
import circuit_util as cu
import tensor_network_util as tnu
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager, passes
from qiskit.transpiler.passes import Unroller, UnrollCustomDefinitions


def get_circuit_setup(circuit: QuantumCircuit) -> QuantumCircuit:
    bench_circ = prepare_circuit(circuit)
    bench_equiv_circ = get_combined_inverse_circuit(bench_circ)
    pm = get_unroll_manager()
    return pm.run(bench_equiv_circ)

def get_circuit_setup_quimb(circuit: QuantumCircuit) -> Circuit:
    return cu.qiskit_to_quimb_circuit(get_circuit_setup(circuit))

def prepare_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    return circuit.remove_final_measurements(inplace=False)

def get_combined_inverse_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    inv_circuit = circuit.inverse()
    return circuit.compose(inv_circuit)

def get_unroll_manager() -> PassManager:
    all_quimb_gates = ['h', 'x', 'y', 'z', 's', 'cx', 'cnot', 'cy', 'cz', 'rz', 'sdg']
    custom_gate_pass_ = Unroller(all_quimb_gates)
    return PassManager(custom_gate_pass_)


if __name__ == "__main__":
    circuit = get_benchmark('dj', 0, 3)
    bench_circ = prepare_circuit(circuit)
    bench_equiv_circ = get_combined_inverse_circuit(bench_circ)
    print(bench_equiv_circ.qasm())

    # bench_quimb = cu.qiskit_to_quimb_circuit(bench_circ)
    # bench_tn = tnu.get_tensor_network(bench_quimb)
    # bench_gate_decomp = tddu.get_tdds_from_quimb_tensor_network(bench_tn)

    print(2)