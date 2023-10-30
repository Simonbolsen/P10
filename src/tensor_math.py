import numpy as np
import tddpure.TDD.TDD as TDD
from tddpure.TDD.TN import Tensor

left = [0.+0.5j, 0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j,  0.+0.j,  0.+0.5j, 0.+0.j,  0.-0.5j, 0.-0.j,  0.-0.5j, 0.-0.j,  0.+0.j,  0.+0.5j,  0.-0.j,  0.-0.5j, 0.+0.j,  0.+0.5j, 0.-0.j,  0.-0.5j, 0.+0.j,  0.+0.5j, 0.-0.j,  0.-0.5j, 0.+0.j,  0.-0.5j, 0.+0.j,  0.+0.5j, 0.+0.5j, 0.+0.j,  0.-0.5j, 0.-0.j,  0.+0.5j, 0.+0.j,  0.-0.5j, 0.-0.j,  0.+0.5j, 0.+0.j,  0.-0.5j, 0.-0.j,  0.-0.5j,  0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j,  0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j, 0.+0.j,  0.+0.5j, 0.-0.j,  0.-0.5j, 0.-0.j,  0.-0.5j]

right = [0.+0.70710678j, 0.+0.j, 0.+0.j, 0.+0.70710678j, 0.+0.70710678j, 0.+0.j, 0.-0.j, 0.-0.70710678j, 0.+0.70710678j, 0.+0.j, 0.+0.j, 0.+0.70710678j, 0.-0.70710678j, 0.+0.j, 0.+0.j, 0.+0.70710678j]

a = np.array(left).reshape((2,2,2,2,2,2))
b = np.array(right).reshape((2,2,2,2))

a_inds = ['k3', 'J', 'R', 'c', 'r', 'w']
b_inds = ['b3', 'r', 'w', 'z']

c_inds = ['b3', 'k3', 'J', 'R', 'c', 'z']

def contract(x, x_inds, y, y_inds, inds):
    all_inds = list(set(x_inds) | set(y_inds))
    t = np.ones([2 for _ in all_inds])
    
    t = insert(t, x, x_inds, all_inds)
    t = insert(t, y, y_inds, all_inds)

    return t

def insert(t, x, x_inds, inds):
    if inds == []:
        return 
    if x_inds[0] == inds[0]:
        return 

def get_indeces(inds):
    return [TDD.Index(i) for i in inds]

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

...