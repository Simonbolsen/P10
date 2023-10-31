import numpy as np
import tddpure.TDD.TDD as TDD
from tddpure.TDD.TN import Tensor
from example import tdd_analysis as example
from correct_example import correct_example
from tdd_util import reverse_lexicographic_key
from tensor_network_util import rectify_complex

def get_indeces(inds):
    return [TDD.Index(i) for i in inds]

def to_complex(data, rounded):
    if isinstance(data[0], list):
        return [to_complex(i) for i in data]
    elif rounded:
        return rectify_complex(data[0] + data[1] * 1j)
    else:
        return data[0] + data[1] * 1j

def get_letter_index(inds, map):
    return "".join([map[i] for i in inds])

def get_einsum_string(data, names, a, b, c, map):
    a_lettes = get_letter_index(data[names[a] + "_inds"], map)
    b_letters = get_letter_index(data[names[b] + "_inds"], map) 
    c_letters = get_letter_index(data[names[c] + "_inds"], map)
    return a_lettes + "," + b_letters + "->" + c_letters

def test_experiment(data, rounded = True):
    names = {"left" : "left_tensor", "right" : "right_tensor", "tensor_result" : "result_tensor", 
             "tdd_result" : "actual_result_tensor", "einsum_result" : "result_tensor"}
    arrays = {name:np.array(to_complex(data[data_key], rounded)) for name, data_key in names.items()}
    global_index_order = sorted(list(set(data["left_tensor_inds"]) | set(data["right_tensor_inds"])), key=reverse_lexicographic_key, reverse=True)
    
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter_map = {index:letters[i] for i,index in enumerate(global_index_order)}
    arrays["einsum_result"] = np.einsum(get_einsum_string(data, names, "left", "right", "einsum_result", letter_map), 
                                        arrays["left"], arrays["right"])

    tensors = {name : Tensor(arrays[name], get_indeces(data[data_key + "_inds"])) for name, data_key in names.items()}
    
    TDD.Ini_TDD(global_index_order)
    tdds = {name : t.tdd() for name, t in tensors.items()}
    tdds["repeated_tdd_result"] = TDD.cont(tdds["left"], tdds["right"])

    print([f"{k}: {i.node_number()}" for k, i in tdds.items()])

    for k, i in tdds.items():
        i.show(name = k + "r" if rounded else "")

def test1():
    left = [0.+0.5j, 0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j,  0.+0.j,  0.+0.5j, 0.+0.j,  0.-0.5j, 0.-0.j,  0.-0.5j, 0.-0.j,  0.+0.j,  0.+0.5j,  0.-0.j,  0.-0.5j, 0.+0.j,  0.+0.5j, 0.-0.j,  0.-0.5j, 0.+0.j,  0.+0.5j, 0.-0.j,  0.-0.5j, 0.+0.j,  0.-0.5j, 0.+0.j,  0.+0.5j, 0.+0.5j, 0.+0.j,  0.-0.5j, 0.-0.j,  0.+0.5j, 0.+0.j,  0.-0.5j, 0.-0.j,  0.+0.5j, 0.+0.j,  0.-0.5j, 0.-0.j,  0.-0.5j,  0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j,  0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j, 0.-0.j,  0.-0.5j, 0.-0.j,  0.-0.5j]
    right = [0.+0.70710678j, 0.+0.j, 0.+0.j, 0.+0.70710678j, 0.+0.70710678j, 0.+0.j, 0.-0.j, 0.-0.70710678j, 0.+0.70710678j, 0.+0.j, 0.+0.j, 0.+0.70710678j, 0.-0.70710678j, 0.+0.j, 0.+0.j, 0.+0.70710678j]

    a = np.array(left).reshape((2,2,2,2,2,2))
    b = np.array(right).reshape((2,2,2,2))

    a_inds = ['k3', 'J', 'R', 'c', 'r', 'w']
    b_inds = ['b3', 'r', 'w', 'z']
    c_inds = ['b3', 'k3', 'J', 'R', 'c', 'z']
    c = np.einsum("abcdef,gefh->gabcdh", a, b)

    TDD.Ini_TDD(['b3', 'k3', 'J', 'R', 'c', 'r', 'w', 'z'])
    ta = Tensor(a, get_indeces(a_inds))
    tb = Tensor(b, get_indeces(b_inds))
    tc = Tensor(c, get_indeces(c_inds))

    tdds = {"a" : ta.tdd(), "b": tb.tdd(), "c" : tc.tdd()}
    tdds["d"] = TDD.cont(tdds["a"], tdds["b"])


    print([f"{k}: {i.node_number()}" for k, i in tdds.items()])

    #for k, i in tdds.items():
    #    i.show(name = k)

def compare_breakpoints():
    first = {}

    breakpoints = {} #file with break point data

    for s in breakpoints.first:
        if s in first:
            first[s] += 1
        else:
            first[s] = 1

    second = {}

    for s in breakpoints.second:
        if s in second:
            second[s] += 1
        else:
            second[s] = 1

    for k in list(set(first.keys()) | set(second.keys())):
        print(f"{k}: {first[k]}, {second[k]}")

if __name__ == "__main__":
    test_experiment(example, True)
    test_experiment(correct_example, True)
