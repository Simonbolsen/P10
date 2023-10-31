import numpy as np
import tddpure.TDD.TDD as TDD
from tddpure.TDD.TN import Tensor
from P9.src.example import tdd_analysis as example

def get_indeces(inds):
    return [TDD.Index(i) for i in inds]

def to_complex(data):
    if isinstance(data[0], list):
        return [to_complex(i) for i in data]
    else:
        return data[0] + data[1] * 1j

def test_experiment(data):
    names = {"left" : "left_tensor", "right" : "right_tensor", "tensor_result" : "result_tensor", "tdd_result" : "actual_result_tensor"}
    arrays = {name:np.array(to_complex(data[data_key])) for name, data_key in names.items()}
    TDD.Ini_TDD(list(set(data["left_tensor_inds"]) | set(data["right_tensor_inds"])))

    tensors = {name : Tensor(arrays[name], get_indeces(data[data_key + "_inds"])) for name, data_key in names.items()}
    tdds = {name : t.tdd() for name, t in tensors.items()}

    print([f"{k}: {i.node_number()}" for k, i in tdds.items()])

    #for k, i in tdds.items():
    #    i.show(name = k)

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

    for k, i in tdds.items():
        i.show(name = k)

if __name__ == "__main__":
    test_experiment(example)