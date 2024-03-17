import ctypes
import sys
import os 
import bench_util as bu
import cotengra as ctg
import os
import tensor_network_util as tnu
import circuit_util as cu
from tdd_util import get_tdds_from_quimb_tensor_network, reverse_lexicographic_key
from first_experiment import contract_tdds, fast_contract_tdds
from tddpure.TDD.TDD import Ini_TDD, TDD, tdd_2_np, cont


class CPPHandler():
    def __init__(self):
        self.debug = False
        self.res_name = "tddRes"
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.handle = ctypes.CDLL(self.dir_path + "/libTDDLinux.so")

        self.handle.pyContractCircuit.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool] 
        self.handle.pyContractCircuit.restype = ctypes.c_char_p

    def CPPContraction(self, circuit, qubits, plan):
        circuit = circuit.replace(",)", ")")
        res = self.handle.pyContractCircuit(str.encode(circuit), qubits, str.encode(plan), str.encode(self.res_name), self.debug)    
        return self.parse_result_string(res.decode('utf-8'))

    def parse_result_string(self, res):
        parts = res.replace(" ", "").split(";")
        return {
            "equivalence": "true" in parts[0],
            "cont_time": int(parts[1])
        }

    def enable_debug(self):
        self.debug = True

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

def load_num_prec_file():
    import json
    file_path = os.path.join("prec_table.json")
    with open(file_path, 'r') as json_file:
        data = json_file.read()
    return json.loads(data)

def replace_low_prec_nums(circuit):
    import re
    table = load_num_prec_file()

    pattern = re.compile("(\\((-?)(\\d+.\\d+),?\\))")
    matches = re.findall(pattern, circuit)

    for orig_str, sign, value in matches:
        if value[:8] in table:
            new_val = table[value[:8]]
            replacement = f"({sign}{new_val})"
            circuit = circuit.replace(orig_str, replacement)

    return circuit

if __name__ == '__main__':
    #generate_number_prec_file()
    #TestCPPFunc(7)
    cpp = CPPHandler()
    cpp.enable_debug()

    settings = {
        "simulate": False,
        "algorithm": "ghz",
        "level": (0, 2),
        "qubits": 2,
        "random_gate_deletions": 0
    }
    data = {
        "settings": settings,
        "circuit_settings": settings,
        "path_settings": {
            "use_proportional": True,
            "model_name":"model_8"
        },
        "path_data": {},
        "circuit_data": {

        },
        "make_dataset": False
    }
    

    #circuit = bu.get_gauss_random_circuit(settings["qubits"])#
    #circuit = bu.get_dual_circuit_setup_quimb(data, draw=False)#cu.get_example_circuit(settings["qubits"])#bu.get_dual_circuit_setup_quimb(data, draw=False)
    circuit = cu.qiskit_to_quimb_circuit(cu.get_simple_equiv_circuit())

    #new_circ = replace_low_prec_nums(cu.quimb_to_qiskit_circuit(circuit))

    tensor_network = tnu.get_tensor_network(circuit, split_cnot=False, state = None)
    #tensor_network.draw()
    #tdd_predict = tnu.get_tdd_path(tensor_network, data)
    stats = {"agree_right": 0, "agree_wrong": 0,"disagree": 0, "python_wrong": 0, "cpp_wrong": 0}
    for i in range(1):
        rgreedy = tnu.get_usable_path(tensor_network, tensor_network.contraction_path(
            ctg.HyperOptimizer(methods = "random-greedy", minimize="flops", max_repeats=100, max_time=60, progbar=True, parallel=False)))
        
        res = cpp.fast_contraction(circuit, tensor_network, rgreedy)
        #print(res)

        print("Running Python\n\n")
        variable_order = sorted(list(tensor_network.all_inds()), key=reverse_lexicographic_key, reverse=True)
        Ini_TDD(variable_order, max_rank=len(variable_order)+1)

        gate_tdds = get_tdds_from_quimb_tensor_network(tensor_network, False)
        data["path"] = rgreedy
        result_tdd = contract_tdds(gate_tdds, data, save_intermediate_results=True, comprehensive_saving=True)
        #result_tdd = fast_contract_tdds(gate_tdds, data)
        #print(f"Equivalent: {data['equivalence']}")

        if not data['equivalence'] or not res["equivalence"]:
            if not data['equivalence'] and not res["equivalence"]:
                print('\033[31m' + "Both are wrong" + '\033[m')
                stats["agree_wrong"] += 1
                result_tdd.show(name=f"pythonTDDRes_{i}")
            elif not data['equivalence']:
                print('\033[31m' + "Python TDD wrong" + '\033[m')
                stats["disagree"] += 1
                stats["python_wrong"] += 1
                #result_tdd.show(name=f"pythonTDDRes_{i}")
            else:
                print('\033[31m' + "C++ TDD wrong" + '\033[m')
                stats["disagree"] += 1
                stats["cpp_wrong"] += 1
        else:
            stats["agree_right"] += 1

    print(stats)


