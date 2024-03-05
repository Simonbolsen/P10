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

charptr = ctypes.POINTER(ctypes.c_char)
handle.pyContractCircuit.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p] 
handle.pyContractCircuit.restype = ctypes.c_char_p

def plan_to_str(plan):

    entry_strs = [f"({val[0]},{val[1]})" for val in plan]
    res = ';'.join(entry_strs)
    return res

def CPPContraction(circuit, qubits, plan):
    res = handle.pyContractCircuit(ctypes.c_char_p(str.encode(circuit)), ctypes.c_int(qubits), ctypes.c_char_p(str.encode(plan)))    
    return res


if __name__ == '__main__':

    settings = {
        "simulate": False,
        "algorithm": "ghz",
        "level": (0, 2),
        "qubits": 5,
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
    circuit = bu.get_dual_circuit_setup_quimb(data, draw=False)
    #circuit = get_circuit(5)

    tensor_network = tnu.get_tensor_network(circuit, split_cnot=True, state = None)
    #tensor_network.draw()

    rgreedy = tnu.get_usable_path(tensor_network, tensor_network.contraction_path(
        ctg.HyperOptimizer(methods = "random-greedy", minimize="flops", max_repeats=1, max_time=60, progbar=False, parallel=False)))
        
    res = CPPContraction(cu.quimb_to_qiskit_circuit(circuit), circuit.N, plan_to_str(rgreedy))
    print(res)
