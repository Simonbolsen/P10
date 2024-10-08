import ctypes
import os 
import torch
import circuit_util as cu
import file_util as fu

class CPPHandler():
    def __init__(self):
        self.debug = False
        self.draw_result = False
        self.make_data = False
        self.res_name = "tddRes2"
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.handle = ctypes.CDLL(self.dir_path + "/libTDDLinux.so")

        self.handle.pyContractCircuit.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool] 
        self.handle.pyContractCircuit.restype = ctypes.c_char_p

        self.handle.pyTestNNModel.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p] 
        self.handle.pyTestNNModel.restype = ctypes.c_char_p        
        
        self.handle.pyTestGraph.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p] 
        self.handle.pyTestGraph.restype = ctypes.c_char_p

        self.handle.pyTestOnlinePlanning.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p] 
        self.handle.pyTestOnlinePlanning.restype = ctypes.c_char_p

        self.handle.pyTestWindowedPlanning.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int] 
        self.handle.pyTestWindowedPlanning.restype = ctypes.c_char_p

        self.handle.pyWindowedPlanning.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_int, ctypes.c_bool] 
        self.handle.pyWindowedPlanning.restype = ctypes.c_char_p

        self.handle.pyLookAheadPlanning.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool] 
        self.handle.pyLookAheadPlanning.restype = ctypes.c_char_p

        self.handle.pyQueuePlanning.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool]
        self.handle.pyQueuePlanning.restype = ctypes.c_char_p

        self.handle.pySetPrecision.argtypes = [ctypes.c_int] 
        self.handle.pySetPrecision.restype = ctypes.c_bool

    def set_precision(self, precision=18):
        if self.handle.pySetPrecision(precision):
            print(f"Changed precision to: {precision}")
        else:
            print(f"Failed to change precision")

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
    
    def WindowedPlanningContraction(self, model_name, circuit, tensor_network, length_indifferent, window_size, parallel):
        edges = self.plan_to_str(self.extract_edges_from_tn(tensor_network))
        new_circ = self.interject_tensor_indices_into_circuit(circuit, tensor_network)
        new_circ = new_circ.replace(",)", ")").replace("|", "#")
        res = self.handle.pyWindowedPlanning(str.encode(new_circ), circuit.N, str.encode(model_name), str.encode(edges), length_indifferent, window_size, parallel)
        res = res.decode('utf-8')
        
        return res

    def windowed_contraction(self, model_name, circuit, tensor_network, length_indifferent=False, window_size=4, parallel=False):
        res = self.WindowedPlanningContraction(model_name, circuit, tensor_network, length_indifferent, window_size, parallel)

        json = self.load_contents_of_temp_file()

        path = [(v[0], v[1]) for v in json["executed_plan"]]
        predicted_sizes = [v[2] for v in json["executed_plan"]]
        actual_sizes = {}
        for step in json["executed_plan"]:
            key = str(step[1])
            if not key in actual_sizes:
                actual_sizes[key] = []
            actual_sizes[key].append(step[3])

        res_data = self.parse_result_string(res)
        res_data["path"] = path
        res_data["pred_sizes"] = predicted_sizes
        res_data["sizes"] = actual_sizes
        res_data["time_data"] = json["time_data"]
        res_data["executed_plan"] = json["executed_plan"]

        return res_data

    def LookAheadContraction(self, circuit, tensor_network, length_indifferent):
        edges = self.plan_to_str(self.extract_edges_from_tn(tensor_network))
        new_circ = self.interject_tensor_indices_into_circuit(circuit, tensor_network)
        new_circ = new_circ.replace(",)", ")").replace("|", "#")
        res = self.handle.pyLookAheadPlanning(str.encode(new_circ), circuit.N, str.encode(edges), str.encode(self.res_name), length_indifferent, self.draw_result)
        res = res.decode('utf-8')
        
        return res

    def look_ahead_contraction(self, circuit, tensor_network, length_indifferent=False):
        res = self.LookAheadContraction(circuit, tensor_network, length_indifferent)

        json = self.load_contents_of_temp_file()
        if not "planning" in json["time_data"]:
            json["time_data"]["planning"] = [0]

        path = [(v[0], v[1]) for v in json["executed_plan"]]
        actual_sizes = {}
        for step in json["executed_plan"]:
            key = str(step[1])
            if not key in actual_sizes:
                actual_sizes[key] = []
            actual_sizes[key].append(step[2])

        res_data = self.parse_result_string(res)
        res_data["path"] = path
        res_data["pred_sizes"] = []
        res_data["sizes"] = actual_sizes
        res_data["time_data"] = json["time_data"]
        res_data["executed_plan"] = json["executed_plan"]

        return res_data
    
    def QueuePlanningContraction(self, circuit, tensor_network, length_indifferent):
        edges = self.plan_to_str(self.extract_edges_from_tn(tensor_network))
        new_circ = self.interject_tensor_indices_into_circuit(circuit, tensor_network)
        new_circ = new_circ.replace(",)", ")").replace("|", "#")
        res = self.handle.pyQueuePlanning(str.encode(new_circ), circuit.N, str.encode(edges), str.encode(self.res_name), length_indifferent, self.draw_result)
        res = res.decode('utf-8')
        
        return res

    def queue_planning_contraction(self, circuit, tensor_network, length_indifferent=False):
        res = self.QueuePlanningContraction(circuit, tensor_network, length_indifferent)

        json = self.load_contents_of_temp_file()
        if not "planning" in json["time_data"]:
            json["time_data"]["planning"] = [0]

        path = [(v[0], v[1]) for v in json["executed_plan"] if v[0] != v[1]]
        actual_sizes = {}
        for step in json["executed_plan"]:
            key = str(step[1])
            if not key in actual_sizes:
                actual_sizes[key] = []
            actual_sizes[key].append(step[2])

        res_data = self.parse_result_string(res)
        res_data["path"] = path
        res_data["pred_sizes"] = []
        res_data["sizes"] = actual_sizes
        res_data["time_data"] = json["time_data"]
        res_data["executed_plan"] = json["executed_plan"]

        return res_data

    def CPPContraction(self, circuit, qubits, plan, length_indifferent=False, expect_equivalence=False):
        circuit = circuit.replace(",)", ")").replace("|", "#")
        res = self.handle.pyContractCircuit(str.encode(circuit), qubits, str.encode(plan), str.encode(self.res_name), length_indifferent, self.debug, self.draw_result, self.make_data, expect_equivalence)    
        return self.parse_result_string(res.decode('utf-8'))

    def load_contents_of_temp_file(self):
        path = os.path.join("..", "temporary_files", "temp_file_for_run.json")
        json = fu.load_json(path)
        #os.remove(path)
        
        return json

    def parse_result_string(self, res):
        parts = res.replace(" ", "").split(";")
        return {
            "equivalence": "true" in parts[0],
            "cont_time": int(parts[1])
        }

    def enable_debug(self):
        self.debug = True

    def data_creation_mode(self):
        self.make_data = False

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

    def fast_contraction(self, circuit, tensor_network, plan, length_indifferent = False, expect_equivalence = False):
        new_circ = self.interject_tensor_indices_into_circuit(circuit, tensor_network)
        #off_plan = self.offset_plan(plan, tensor_network.tensor_map, split_cnot)
        plan_str = self.plan_to_str(plan)
        res = self.CPPContraction(new_circ, circuit.N, plan_str, length_indifferent=length_indifferent, expect_equivalence=expect_equivalence)
        
        json = self.load_contents_of_temp_file()
        json["time_data"]["planning"] = [0]

        path = [(v[0], v[1]) for v in json["executed_plan"]]
        actual_sizes = {}
        for step in json["executed_plan"]:
            key = str(step[1])
            if not key in actual_sizes:
                actual_sizes[key] = [0]
            actual_sizes[key].append(step[2])

        res["executed_plan"] = json["executed_plan"]
        res["path"] = path
        res["pred_sizes"] = []
        res["sizes"] = actual_sizes
        res["time_data"] = json["time_data"]

        return res
