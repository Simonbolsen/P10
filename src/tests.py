from efficient_contraction import CPPHandler
import tensor_network_util as tnu
import bench_util as bu

class CPPContractionTests():
    def __init__(self):
        self.debug = False
        self.cpp_handler = CPPHandler()
        self.quiet = False
    
    def test_equivalent_circuits_with_linear_on_all_algorithms(self):
        algorithms = ["dj", "ghz", "graphstate", "wstate", "qftentangled", "twolocalrandom", "qpeexact", "realamprandom", "su2random"]
        for algorithm in algorithms:
            self.test_equivalent_circuits_with_linear_on_one_algorithm(algorithm)

    def test_equivalent_circuits_with_linear_on_one_algorithm(self, algorithm):
        settings = {
            "simulate": False,
            "algorithm": algorithm,
            "level": (0, 2),
            "qubits": 4,
            "random_gate_deletions": 0
        }
        data = {
            "settings": settings,
            "circuit_settings": settings,
            "path_settings": {"use_proportional": False},
            "path_data": {},
            "circuit_data": {},
            "make_dataset": False
        }
        
        circuit = bu.get_dual_circuit_setup_quimb(data, draw=False)
        tensor_network = tnu.get_tensor_network(circuit, split_cnot=False, state = None)

        path = tnu.get_linear_path(tensor_network, 0.0, False)
        res = self.cpp_handler.fast_contraction(circuit, tensor_network, path)
        
        if not self.quiet:
            if res["equivalence"]:
                print('\033[92m' + f"Succeed with CPP contraction with linear plan on {algorithm}" + '\033[m')
            else:
                print('\033[31m' + f"Failed with CPP contraction with linear plan on {algorithm}" + '\033[m')

        return res["equivalence"]

    def test_all(self):
        success, fails = self.test_equivalent_circuits()


if __name__ == '__main__':
    tester = CPPContractionTests()
    tester.test_equivalent_circuits_with_linear_on_all_algorithms()