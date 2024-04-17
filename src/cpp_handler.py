import ctypes
import os 
import torch
import circuit_util as cu

class CPPHandler():
    def __init__(self):
        self.debug = False
        self.draw_result = False
        self.make_data = False
        self.res_name = "tddRes2"
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.handle = ctypes.CDLL(self.dir_path + "/libTDDLinux.so")

        self.handle.pyContractCircuit.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_int] 
        self.handle.pyContractCircuit.restype = ctypes.c_char_p

        self.handle.pyTestNNModel.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p] 
        self.handle.pyTestNNModel.restype = ctypes.c_char_p        
        
        self.handle.pyTestGraph.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p] 
        self.handle.pyTestGraph.restype = ctypes.c_char_p

        self.handle.pyTestOnlinePlanning.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p] 
        self.handle.pyTestOnlinePlanning.restype = ctypes.c_char_p

        self.handle.pyTestWindowedPlanning.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int] 
        self.handle.pyTestWindowedPlanning.restype = ctypes.c_char_p

    def TestNNModel(self, model_name, circuit, tensor_network, plan):
        new_circ = self.interject_tensor_indices_into_circuit(circuit, tensor_network)
        plan_str = self.plan_to_str(plan)
        new_circ = new_circ.replace(",)", ")").replace("|", "#")
        res = self.handle.pyTestNNModel(str.encode(model_name), str.encode(new_circ), circuit.N, str.encode(plan_str))
        return res.decode('utf-8')

    def TestGraph(self, model_name, circuit, tensor_network, plan):
        edges = self.plan_to_str(self.extract_edges_from_tn(tensor_network))
        new_circ = self.interject_tensor_indices_into_circuit(circuit, tensor_network)
        plan_str = self.plan_to_str(plan)
        new_circ = new_circ.replace(",)", ")").replace("|", "#")
        res = self.handle.pyTestGraph(str.encode(model_name), str.encode(new_circ), circuit.N, str.encode(plan_str), str.encode(edges))
        return res.decode('utf-8')
    
    def TestOnlinePlanning(self, model_name, circuit, tensor_network):
        edges = self.plan_to_str(self.extract_edges_from_tn(tensor_network))
        new_circ = self.interject_tensor_indices_into_circuit(circuit, tensor_network)
        new_circ = new_circ.replace(",)", ")").replace("|", "#")
        res = self.handle.pyTestOnlinePlanning(str.encode(new_circ), circuit.N, str.encode(model_name), str.encode(edges))
        return res.decode('utf-8')

    def TestWindowedPlanning(self, model_name, circuit, tensor_network, window_size=4):
        edges = self.plan_to_str(self.extract_edges_from_tn(tensor_network))
        new_circ = self.interject_tensor_indices_into_circuit(circuit, tensor_network)
        new_circ = new_circ.replace(",)", ")").replace("|", "#")
        res = self.handle.pyTestWindowedPlanning(str.encode(new_circ), circuit.N, str.encode(model_name), str.encode(edges), window_size)
        return res.decode('utf-8')

    def CPPContraction(self, circuit, qubits, plan, length_indifferent=False, expect_equivalence=False, precision=18):
        circuit = circuit.replace(",)", ")").replace("|", "#")
        res = self.handle.pyContractCircuit(str.encode(circuit), qubits, str.encode(plan), str.encode(self.res_name), length_indifferent, self.debug, self.draw_result, self.make_data, expect_equivalence, precision)    
        return self.parse_result_string(res.decode('utf-8'))

    def parse_result_string(self, res):
        parts = res.replace(" ", "").split(";")
        return {
            "equivalence": "true" in parts[0],
            "cont_time": int(parts[1])
        }

    def enable_debug(self):
        self.debug = True

    def data_creation_mode(self):
        self.make_data = True

    def show_result(self):
        self.draw_result = True

    def plan_to_str(self, plan):

        entry_strs = [f"({val[0]},{val[1]})" for val in plan]
        res = ';'.join(entry_strs)
        return res

    def offset_plan(self, plan, tensor_map, split_cnot):
        def get_gate_num_from_tag(tags):
            for tag in tags:
                if "GATE" in tag:
                    return int(tag.split('_')[1])
            return -1
        return [(get_gate_num_from_tag(tensor_map[val[0]].tags), get_gate_num_from_tag(tensor_map[val[1]].tags)) for val in plan]

    def extract_edges_from_tn(self, tn):
        res = []
        for _, val in tn.ind_map.items():
            val = list(val)
            if len(val) < 2:
                continue
            res.append((val[1], val[0]) if val[0] > val[1] else val)

        return res 

    """
    gate_tag = [tag for tag in tensor.tags if re.match(r"GATE_\d+$", tag)][0]
    other_tensor = tn.tensor_map[[v for v in list(tn.tag_map[gate_tag]) if v != ent][0]]
    common_idx = list(set(tensor.inds) & set(other_tensor.inds))[0]
    str_pair = tuple([ind for ind in tensor.inds if ind != common_idx])
    if list(tensor.inds).index(common_idx) > list(other_tensor.inds).index(common_idx):
        in_btw_qubit = max(int([tag for tag in tensor.tags if re.match(r"I\d+$", tag)][0][1:]), int([tag for tag in other_tensor.tags if re.match(r"I\d+$", tag)][0][1:]))
        in_between_lists[in_btw_qubit-1].append(common_idx)

    """
    def interject_tensor_indices_into_circuit(self, quimb_circuit, tn):
        gate_strs = []
        split_offset = 0
        for i, g in enumerate(quimb_circuit.gates):
            gate_name = g.label.lower()
            tensor_idxs = list(tn.tag_map[f"GATE_{i}"])
            if gate_name in ["cx", "cy", "cz"] and len(tensor_idxs) > 1:
                common_idx = list(set(tn.tensor_map[tensor_idxs[0]].inds) & set(tn.tensor_map[tensor_idxs[1]].inds))[0]
                control_tensor_idx = tensor_idxs[0] if list(tn.tensor_map[tensor_idxs[0]].inds).index(common_idx) > list(tn.tensor_map[tensor_idxs[1]].inds).index(common_idx) else tensor_idxs[1]
                target_tensor_idx = [t for t in tensor_idxs if t != control_tensor_idx][0]
                gate_strs.append(f"{str(control_tensor_idx)}|{cu.handle_gate_str(g).replace(gate_name, gate_name + '_c')}")
                gate_strs.append(f"{str(target_tensor_idx)}|{cu.handle_gate_str(g).replace(gate_name, gate_name + '_t')}")
            else:
                gate_strs.append(f"{str(tensor_idxs[0])}|{cu.handle_gate_str(g)}")

        return cu.get_qasm_header(quimb_circuit.N) + '\n'.join(gate_strs)

    def fast_contraction(self, circuit, tensor_network, plan, length_indifferent = False, expect_equivalence = False, precision = 18):
        new_circ = self.interject_tensor_indices_into_circuit(circuit, tensor_network)
        #off_plan = self.offset_plan(plan, tensor_network.tensor_map, split_cnot)
        plan_str = self.plan_to_str(plan)
        return self.CPPContraction(new_circ, circuit.N, plan_str, length_indifferent=length_indifferent, expect_equivalence=expect_equivalence, precision=precision)
    
