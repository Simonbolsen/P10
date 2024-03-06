import ctypes
import sys
import os 
import bench_util as bu
import cotengra as ctg
import os
import tensor_network_util as tnu
import circuit_util as cu

dir_path = os.path.dirname(os.path.realpath(__file__))
handle = ctypes.CDLL(dir_path + "/libTDDLinux.so")
print(handle.pyContractCircuit) 

charptr = ctypes.POINTER(ctypes.c_char)
handle.pyContractCircuit.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p] 
handle.pyContractCircuit.restype = ctypes.c_char_p


handle.testerFunc.argtypes = [ctypes.c_int] 
handle.testerFunc.restype = ctypes.c_int


def plan_to_str(plan):

    entry_strs = [f"({val[0]},{val[1]})" for val in plan]
    res = ';'.join(entry_strs)
    return res

def offset_plan(plan, tensor_map):
    def get_gate_num_from_tag(tags):
        for tag in tags:
            if "GATE" in tag:
                gate_num = int(tag.split('_')[1])
                return gate_num

        return -1

    return [(get_gate_num_from_tag(tensor_map[val[0]].tags), get_gate_num_from_tag(tensor_map[val[1]].tags)) for val in plan]

def CPPContraction(circuit, qubits, plan):
    res = handle.pyContractCircuit(str.encode(circuit), qubits, str.encode(plan))    
    return res

def TestCPPFunc(num):
    res = handle.testerFunc(num)
    print(res)

if __name__ == '__main__':

    TestCPPFunc(7)

    settings = {
        "simulate": False,
        "algorithm": "dj",
        "level": (0, 2),
        "qubits": 13,
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
    circuit = bu.get_gauss_random_circuit(settings["qubits"])#bu.get_dual_circuit_setup_quimb(data, draw=False)#cu.get_example_circuit(settings["qubits"])#bu.get_dual_circuit_setup_quimb(data, draw=False)
    #circuit = get_circuit(5)

    tensor_network = tnu.get_tensor_network(circuit, split_cnot=False, state = None)
    #tensor_network.draw()

    rgreedy = tnu.get_usable_path(tensor_network, tensor_network.contraction_path(
        ctg.HyperOptimizer(methods = "random-greedy", minimize="flops", max_repeats=1, max_time=60, progbar=False, parallel=False)))
        
    res = CPPContraction(cu.quimb_to_qiskit_circuit(circuit), circuit.N, plan_to_str(offset_plan(rgreedy, tensor_network.tensor_map)))
    print(res)
