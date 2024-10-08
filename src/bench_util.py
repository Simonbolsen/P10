from mqt.bench import CompilerSettings, QiskitSettings, TKETSettings
from mqt.bench import get_benchmark
from mqt.bench.utils import get_supported_benchmarks
from mqt import qcec
import mqt.bench.qiskit_helper as qiskit_helper
from quimb.tensor import Circuit
from quimb.tensor.circuit import Gate
import tdd_util as tddu
import circuit_util as cu
from circuit_util import all_quimb_gates
import tensor_network_util as tnu
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Operator
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager, passes
from qiskit.transpiler.passes import UnrollCustomDefinitions, Decompose
from random import randint, random, choice, gauss
import numpy as np
from math import floor
from unroller import Unroller
from mqt.bench.devices import IBMProvider


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

def sanity_check(circuit: QuantumCircuit, data):
    identity_qc = QuantumCircuit(data["circuit_settings"]["qubits"])
    
    au = Operator(circuit)
    bu = Operator(identity_qc)
    data["sanity_check"] = au.dim == bu.dim and np.allclose(au.data, bu.data)

def get_dual_circuit_setup(c1: QuantumCircuit, c2: QuantumCircuit, data, draw: bool = False) -> QuantumCircuit:
    bench_circ1 = prepare_circuit(c1)
    bench_circ2 = prepare_circuit(c2)
    if draw:
        print(bench_circ1.count_ops().values())
        print(bench_circ2.count_ops().values())
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

    #sanity_check(unrolled_circ, data)

    bench_circ1_copy = bench_circ1.copy()
    # Find start of second circuit:
    unrolled_first_circ_gate_count = sum(pm.run(bench_circ1_copy).count_ops().values())
    data["circuit_data"]["unrolled_first_circ_gate_count"] = unrolled_first_circ_gate_count
    if draw:
        print(f"Second circuit is of size: {sum(pm.run(bench_circ2.copy()).count_ops().values())}")
    if data["path_settings"]["use_proportional"]:
        data["path_settings"]["linear_fraction"] = unrolled_first_circ_gate_count / data["circuit_data"]["unrolled_qiskit_gate_count"]

    if data["circuit_settings"]["random_gate_deletions"] > (data["circuit_data"]["unrolled_qiskit_gate_count"] - unrolled_first_circ_gate_count + 1):
        # Deleting all gates is not allowed
        raise ValueError("Trying to delete too many gates")
    
    data["circuit_data"]["random_gate_deletions"] = []
    for i in range(data["circuit_settings"]["random_gate_deletions"]):
        # Randomly delete gates
        random_gate_index = randint(unrolled_first_circ_gate_count, data["circuit_data"]["unrolled_qiskit_gate_count"] - (i + 1))
        data["circuit_data"]["random_gate_deletions"].append(random_gate_index)
        del unrolled_circ.data[random_gate_index]

    return unrolled_circ

def get_circuit_setup_quimb(circuit: QuantumCircuit, draw: bool = False) -> Circuit:
    return cu.qiskit_to_quimb_circuit(get_circuit_setup(circuit, draw))

level_mapping = {
    0: lambda qc: qc.copy(),
    1: lambda qc: get_independent_level(qc),
    2: lambda qc: get_native_gates_level(qc),
    3: lambda qc: get_mapped_level(qc)
}

def get_rounded_circuit(circuit: QuantumCircuit, decimal_places=4) -> QuantumCircuit:
    if decimal_places < 0:
        return circuit
    
    import re
    rounder = re.compile(r"\d*\.\d+")
    def mround(match):
        return "{:.{prec}f}".format(float(match.group()), prec=decimal_places)

    rounded_qasm_circuit = re.sub(rounder, mround, circuit.qasm())
    return QuantumCircuit.from_qasm_str(rounded_qasm_circuit)

def get_dual_circuit_setup_quimb(data, draw: bool = False, as_qiskit: bool= False) -> Circuit:
    assert data["circuit_settings"] is not None
    circ_conf = data["circuit_settings"]

    assert circ_conf["algorithm"] is not None and circ_conf["level"] is not None and circ_conf["qubits"] is not None
    
    base_circ = get_rounded_circuit(get_benchmark(circ_conf["algorithm"], level=0, circuit_size=circ_conf["qubits"]), decimal_places=-1)

    c1 = level_mapping[circ_conf["level"][0]](base_circ)
    c2 = level_mapping[circ_conf["level"][1]](base_circ)
    
    data["circuit_data"]["circuit_1_qasm"] = c1
    data["circuit_data"]["circuit_2_qasm"] = c2

    # c1 = get_benchmark(circ_conf["algorithm"], circ_conf["level"][0], circ_conf["qubits"])
    # c2 = get_benchmark(circ_conf["algorithm"], circ_conf["level"][1], circ_conf["qubits"])
    return get_dual_circuit_setup_quimb_from_circuits(c1, c2, data, draw, as_qiskit)

def get_circuit_from_file(file):
    return QuantumCircuit.from_qasm_file(file)

def get_dual_circuit_setup_from_practical_circuits(data, draw: bool = False, as_qiskit: bool = False) -> Circuit:
    assert data["circuit_settings"] is not None
    circ_conf = data["circuit_settings"]

    file = circ_conf["algorithm_file_path"]
    base_circ = get_rounded_circuit(get_circuit_from_file(file), decimal_places=-1)
    
    c1 = level_mapping[circ_conf["level"][0]](base_circ)
    c2 = level_mapping[circ_conf["level"][1]](base_circ)
    
    data["circuit_data"]["circuit_1_qasm"] = c1
    data["circuit_data"]["circuit_2_qasm"] = c2

    return get_dual_circuit_setup_quimb_from_circuits(c1, c2, data, draw, as_qiskit)

def get_dual_circuit_setup_quimb_from_circuits(c1: QuantumCircuit, c2: QuantumCircuit, data, draw: bool = False, as_qiskit: bool = False) -> Circuit:
    qiskit_circuit = get_dual_circuit_setup(c1, c2, data, draw=draw)
    return qiskit_circuit if as_qiskit else cu.qiskit_to_quimb_circuit(qiskit_circuit)

def prepare_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    return circuit.remove_final_measurements(inplace=False)

def get_combined_inverse_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    inv_circuit = circuit.inverse()
    return circuit.compose(inv_circuit)

def get_unroll_manager() -> PassManager:
    
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


def get_independent_level(circuit: QuantumCircuit) -> QuantumCircuit:
    qc = circuit.copy()
    return qiskit_helper.get_indep_level(qc, num_qubits=None, file_precheck=False, return_qc=True)

def get_native_gates_level(circuit: QuantumCircuit) -> QuantumCircuit:
    qc = circuit.copy()
    # by https://github.com/cda-tum/mqt-bench/blob/3cedf4b76c5773f3e14b434258b53f49eae2877c/src/mqt/bench/benchmark_generator.py#L45
    return qiskit_helper.get_native_gates_level(qc, num_qubits=None, provider=IBMProvider, opt_level=1, file_precheck=False, return_qc=True)

def get_mapped_level(circuit: QuantumCircuit) -> QuantumCircuit:
    qc = circuit.copy()
    return qiskit_helper.get_mapped_level(qc, num_qubits=None, provider=IBMProvider, device_name="ibm_washington", opt_level=1, file_precheck=False, return_qc=True)


def generate_one_qubit_random_gate(on_qubit):
    gate_label = choice(cu.one_qubit_quimb_gates)
    return Gate(gate_label, (-1.5707963267948966,) if gate_label in ["rz", "rx", "ry"] else (), (on_qubit,))

def generate_two_qubit_random_gate(control_qubit, target_qubit):
    return Gate(choice(cu.two_qubit_quimb_gates), (), (control_qubit,target_qubit))

def generate_random_gate(qubits):
    one_qubit_gate_bias = 0.3
    
    target_qubit = randint(0, qubits - 1)
    control_qubit = randint(0, qubits - 1)
    while (control_qubit == target_qubit):
        control_qubit = randint(0, qubits - 1)

    new_gate = generate_two_qubit_random_gate(control_qubit, target_qubit) if random() >= one_qubit_gate_bias else generate_one_qubit_random_gate(target_qubit)

    return new_gate

def mutation_action(qubits, gates, data):
    action = choice(['delete', 'insert', 'change'])
    action_index = randint(0, len(gates) - 1)
    in_first = data["circuit_data"]["unrolled_first_circ_gate_count"] <= action_index + 1 

    if action == 'delete':
        if in_first:
            data["circuit_data"]["unrolled_first_circ_gate_count"] -= 1
        return gates[:action_index] + gates[action_index+1:]
    else:
        new_gate = generate_random_gate(qubits)
        if action == 'insert':
            if in_first:
                data["circuit_data"]["unrolled_first_circ_gate_count"] += 1
            return gates[:action_index] + [new_gate] + gates[action_index:]
        return gates[:action_index] + [new_gate] + gates[action_index+1:]

def add_gate_to_circuit_at_random(qubits, gates):
    action_index = 0 if len(gates) == 0 else randint(0, len(gates) - 1)
    new_gate = generate_random_gate(qubits)
    return gates[:action_index] + [new_gate] + gates[action_index:]

def mutate_circuit(circuit, mutation_degree, data):
    num_of_mutations = int(np.ceil(len(circuit.gates) * mutation_degree))
    gates = circuit.gates

    for _ in range(num_of_mutations):
        gates = mutation_action(circuit.N, gates, data)

    circuit.gates = gates
    return cu.refresh_circuit(circuit)

def one_random_gate_for_each_wire(gates, qubits):
    one_qubit_gate_bias = 0.7
    new_gates = []
    for q in range(qubits):
        control_qubit = randint(0, qubits - 1)
        while (control_qubit == q):
            control_qubit = randint(0, qubits - 1)
        new_gates.append(generate_two_qubit_random_gate(control_qubit, q) if random() >= one_qubit_gate_bias else generate_one_qubit_random_gate(q))
        
    return gates + new_gates

def get_random_circuit(qubits, num_of_gates):
    circuit = Circuit(qubits)
    gates = circuit.gates

    gates = one_random_gate_for_each_wire(gates, qubits)

    for _ in range(num_of_gates):
        gates = add_gate_to_circuit_at_random(circuit.N, gates)

    circuit.gates = gates
    return cu.refresh_circuit(circuit)

def get_gauss_random_circuit(qubits):
    num_of_gates = min(max(qubits, floor(gauss(2*qubits, 1*qubits))), 3*qubits)
    return get_random_circuit(qubits, num_of_gates)

def get_dual_circuit_setup_from_random_circuits(data, draw: bool = False, as_qiskit: bool = False) -> Circuit:
    assert data["circuit_settings"] is not None
    circ_conf = data["circuit_settings"]
    
    rnd_circuit = get_gauss_random_circuit(circ_conf["qubits"])
    base_circ = get_rounded_circuit(cu.quimb_to_qiskit_circuit(rnd_circuit, as_obj=True), decimal_places=-1)
    
    c1 = level_mapping[circ_conf["level"][0]](base_circ)
    c2 = level_mapping[circ_conf["level"][1]](base_circ)
    
    data["circuit_data"]["circuit_1_qasm"] = c1
    data["circuit_data"]["circuit_2_qasm"] = c2

    return get_dual_circuit_setup_quimb_from_circuits(c1, c2, data, draw, as_qiskit)

def get_combined_circuit_example(algorithm='ghz', qubits=5):
    settings = {
        "simulate": False,
        "algorithm": algorithm,
        "level": (0, 2),
        "qubits": qubits,
        "random_gate_deletions": 0
    }
    data = {
        "circuit_settings": settings,
        "path_settings": {
            "use_proportional": True
        },
        "circuit_data": {

        }
    }

    return data, get_dual_circuit_setup_quimb(data, draw=False)


if __name__ == "__main__":
    circ = get_combined_circuit_example()
    mutate_circuit(circ, 0.4)
    print("fd")
    
    # circuit = get_benchmark('dj', "alg", 3)
    # bench_circ = get_circuit_setup_quimb(circuit)

    # bench_tn = tnu.get_tensor_network(bench_circ, include_state=False, split_cnot=False)
    # bench_gate_decomp = tddu.get_tdds_from_quimb_tensor_network(bench_tn)

    # tddu.draw_all_tdds(bench_gate_decomp)

    # print(2)