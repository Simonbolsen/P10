import ctypes
import os 
import os
import circuit_util as cu

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