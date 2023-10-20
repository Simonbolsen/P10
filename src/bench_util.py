from mqt.bench import CompilerSettings, QiskitSettings, TKETSettings
from mqt.bench import get_benchmark
from mqt.bench.utils import get_supported_benchmarks
from quimb.tensor import Circuit
import tdd_util as tddu
import circuit_util as cu
import tensor_network_util as tnu
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager, passes
from qiskit.transpiler.passes import Unroller, UnrollCustomDefinitions, Decompose

selected_algorithms = [
    "dj",           # smaller
    "graphstate",   # useful
    "qaoa",         # complex
    "vqe",          # useful
]


def get_circuit_setup(circuit: QuantumCircuit, draw: bool = False) -> QuantumCircuit:
    bench_circ = prepare_circuit(circuit)
    if draw:
        print(bench_circ)
    bench_equiv_circ = get_combined_inverse_circuit(bench_circ)
    if draw:
        print(bench_equiv_circ)
    pm = get_unroll_manager()
    unrolled_circ = pm.run(bench_equiv_circ)
    if draw: 
        print(unrolled_circ)
    return unrolled_circ

def get_dual_circuit_setup(c1: QuantumCircuit, c2: QuantumCircuit, data, draw: bool = False) -> QuantumCircuit:
    bench_circ1 = prepare_circuit(c1)
    bench_circ2 = prepare_circuit(c2)
    if draw:
        print(bench_circ1)
        print(bench_circ2)
    comb_circuit = bench_circ1.compose(bench_circ2.inverse())

    data["circuit_data"]["qiskit_gate_count"] = sum(comb_circuit.count_ops().values())
    data["circuit_data"]["qiskit_gate_count_by_type"] = comb_circuit.count_ops()
    data["circuit_data"]["depth"] = comb_circuit.depth()
    data["circuit_data"]["nonlocal_gates"] = comb_circuit.num_nonlocal_gates()

    if draw:
        print(f"Number of (qiskit) gates: {data['circuit_data']['qiskit_gate_count']}")
        print(comb_circuit)
    pm = get_unroll_manager()
    unrolled_circ = pm.run(comb_circuit)
    
    data["circuit_data"]["unrolled_qiskit_gate_count"] = sum(unrolled_circ.count_ops().values())
    data["circuit_data"]["unrolled_qiskit_gate_count_by_type"] = unrolled_circ.count_ops()
    data["circuit_data"]["unrolled_depth"] = unrolled_circ.depth()
    data["circuit_data"]["unrolled_nonlocal_gates"] = unrolled_circ.num_nonlocal_gates()

    if draw: 
        print(unrolled_circ)
    return unrolled_circ

def get_circuit_setup_quimb(circuit: QuantumCircuit, draw: bool = False) -> Circuit:
    return cu.qiskit_to_quimb_circuit(get_circuit_setup(circuit, draw))

def get_dual_circuit_setup_quimb(data, draw: bool = False) -> Circuit:
    assert data["circuit_settings"] is not None
    circ_conf = data["circuit_settings"]

    assert circ_conf["algorithm"] is not None and circ_conf["level"] is not None and circ_conf["qubits"] is not None

    c1 = get_benchmark(circ_conf["algorithm"], circ_conf["level"][0], circ_conf["qubits"])
    c2 = get_benchmark(circ_conf["algorithm"], circ_conf["level"][1], circ_conf["qubits"])
    return get_dual_circuit_setup_quimb_from_circuits(c1, c2, data, draw)

def get_dual_circuit_setup_quimb_from_circuits(c1: QuantumCircuit, c2: QuantumCircuit, data, draw: bool = False) -> Circuit:
    qiskit_circuit = get_dual_circuit_setup(c1, c2, data, draw=draw)
    return cu.qiskit_to_quimb_circuit(qiskit_circuit)

def prepare_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    return circuit.remove_final_measurements(inplace=False)

def get_combined_inverse_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    inv_circuit = circuit.inverse()
    return circuit.compose(inv_circuit)

def get_unroll_manager() -> PassManager:
    all_quimb_gates = ['h', 'x', 'y', 'z', 's', 't', 'cx', 'cnot', 'cy', 'cz', 'rz', 'rx', 'ry' 'sdg', 'tdg', 
                       'x_1_2', 'y_1_2', 'z_1_2', 'w_1_2', 'hz_1_2', 'iswap', 'swap', 'iden', 'u3', 'u2', 'u1',
                       'cu3', 'cu2', 'cu1', 'fsim', 'fsimg', 'givens', 'rxx', 'ryy', 'rzz', 'crx', 'cry', 'crz',
                       'su4', 'ccx', 'ccnot', 'toffoli', 'ccy', 'ccz', 'cswap', 'fredkin', 'u']
    custom_gate_pass_ = Unroller(all_quimb_gates)
    qft_remover = Decompose(gates_to_decompose="QFT")
    qftdg_remover = Decompose(gates_to_decompose="QFT_dg")
    phase_remover = Decompose(gates_to_decompose="P")

    return PassManager([qft_remover, qftdg_remover, phase_remover, custom_gate_pass_])


def generate_testing_set(algorithms: [str], levels: [int], qubits: [int]) -> list[QuantumCircuit]:
    circuits = []
    for algorithm in algorithms:
        for level in levels:
            for qubit in qubits:
                circuits.append(get_benchmark(algorithm, level, qubit))

    return circuits

def quimb_setup_circuit_transform(circuits: [QuantumCircuit]) -> [Circuit]:
    return [get_circuit_setup_quimb(circuit) for circuit in circuits]

def get_benchmark_circuit(config):
    return get_benchmark(config["algorithm"], config["level"], config["qubits"])

def vary_base_algorithm_set(num_of_qubits: int, abstraction_level: int, algorithms: [str] = None) -> list[Circuit]:
    if algorithms is None:
        algorithms = get_supported_benchmarks()

    circuits = [get_benchmark(alg, abstraction_level, num_of_qubits) for alg in algorithms]

    prepared_circuits = [get_circuit_setup_quimb(circuit) for circuit in circuits]
    return prepared_circuits

def vary_abstraction_level_set(num_of_qubits: int, base_algorithm: str) -> list[Circuit]:
    circuits = [get_benchmark(base_algorithm, level, num_of_qubits) for level in range(3)]

    prepared_circuits = [get_circuit_setup_quimb(circuit) for circuit in circuits]
    return prepared_circuits

def vary_number_of_qubits_set(base_algorithm: str, abstraction_level: int, qubit_interval: (int, int) = (1, 10)) -> list[Circuit]:
    qubit_range = range(qubit_interval[0], qubit_interval[1])
    circuits = [get_benchmark(base_algorithm, abstraction_level, num_of_qubits) for num_of_qubits in qubit_range]

    prepared_circuits = [get_circuit_setup_quimb(circuit) for circuit in circuits]
    return prepared_circuits



if __name__ == "__main__":
    circuit = get_benchmark('dj', "alg", 3)
    bench_circ = get_circuit_setup_quimb(circuit)

    bench_tn = tnu.get_tensor_network(bench_circ, include_state=False, split_cnot=False)
    bench_gate_decomp = tddu.get_tdds_from_quimb_tensor_network(bench_tn)

    tddu.draw_all_tdds(bench_gate_decomp)

    print(2)