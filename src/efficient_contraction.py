import ctypes
import sys
import os 
import bench_util as bu
import cotengra as ctg
import os
import tensor_network_util as tnu
import circuit_util as cu


class CPPHandler():
    def __init__(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.handle = ctypes.CDLL(self.dir_path + "/libTDDLinux.so")

        self.handle.pyContractCircuit.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p] 
        self.handle.pyContractCircuit.restype = ctypes.c_char_p

    def CPPContraction(self, circuit, qubits, plan):
        circuit = circuit.replace(",)", ")")
        res = self.handle.pyContractCircuit(str.encode(circuit), qubits, str.encode(plan))    
        return res

    def plan_to_str(self, plan):

        entry_strs = [f"({val[0]},{val[1]})" for val in plan]
        res = ';'.join(entry_strs)
        return res

    def offset_plan(self, plan, tensor_map):
        def get_gate_num_from_tag(tags):
            for tag in tags:
                if "GATE" in tag:
                    return int(tag.split('_')[1])
            return -1
        return [(get_gate_num_from_tag(tensor_map[val[0]].tags), get_gate_num_from_tag(tensor_map[val[1]].tags)) for val in plan]

    def fast_contraction(self, circuit, tensor_network, plan):
        off_plan = self.offset_plan(plan, tensor_network.tensor_map)
        off_plan_str = self.plan_to_str(off_plan)
        return self.CPPContraction(cu.quimb_to_qiskit_circuit(circuit), circuit.N, off_plan_str)



def create_number_prec_table(lb, ub):
    from mpmath import mp
    import numpy as np
    

    def gen_wstate_param(qubits):
        mp.dps = 55
        return {str(np.arccos(np.sqrt(1 / (qubits - k + 1))))[:8]:
                str(mp.acos(mp.sqrt(mp.mpf(1) / mp.mpf(qubits - k + 1))))
                    for k in range(1, qubits)}

    wstate_params = {}
    for i in range(lb, ub):
        wstate_params = dict(wstate_params, **gen_wstate_param(i))

    return wstate_params

def generate_number_prec_file():
    import json
    r = create_number_prec_table(3, 200)
    file_name = "prec_table.json"
    with open(file_name, "w") as file:
        json.dump(r, file, indent=4)


if __name__ == '__main__':

    generate_number_prec_file()

    #TestCPPFunc(7)
    cpp = CPPHandler()

    settings = {
        "simulate": False,
        "algorithm": "graphstate",
        "level": (0, 2),
        "qubits": 256,
        "random_gate_deletions": 0
    }
    data = {
        "circuit_settings": settings,
        "path_settings": {
            "use_proportional": True,
            "model_name":"model_8"
        },
        "path_data": {},
        "circuit_data": {

        }
    }
    

    #circuit = bu.get_gauss_random_circuit(settings["qubits"])#
    circuit = bu.get_dual_circuit_setup_quimb(data, draw=False)#cu.get_example_circuit(settings["qubits"])#bu.get_dual_circuit_setup_quimb(data, draw=False)
    #circuit = get_circuit(5)

    tensor_network = tnu.get_tensor_network(circuit, split_cnot=False, state = None)
    #tensor_network.draw()
    #tdd_predict = tnu.get_tdd_path(tensor_network, data)
    for _ in range(10):
        rgreedy = tnu.get_usable_path(tensor_network, tensor_network.contraction_path(
            ctg.HyperOptimizer(methods = "random-greedy", minimize="flops", max_repeats=100, max_time=60, progbar=True, parallel=False)))
        
        res = cpp.fast_contraction(circuit, tensor_network, rgreedy)
        print(res)
