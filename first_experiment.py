from quimb.tensor import Circuit
import cotengra as ctg
import numpy as np
import random
from tddfork.TDD.TDD import Ini_TDD, TDD
from tddfork.TDD.TN import Index,Tensor,TensorNetwork
from tddfork.TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs
import circuit_util as cu
import tdd_util as tddu
import tensor_network_util as tnu
import bench_util as bu
import os
import datetime

import matplotlib as mpl
mpl.use("TkAgg")


selected_algorithms = [
    "ghz",
    "graphstate",
    "twolocalrandom",
    "qftentangled", # Not working
    "dj",
    "qpeexact", # Not working
    "su2random",
    "wstate",
    "realamprandom"
]


def first_experiment():
    # Prepare save folder and file paths
    folder_path = os.path.join("experiments", f"first_experiment_{datetime.date}")

    # Prepare benchmark circuits:
    bench_circuits = bu.generate_testing_set([selected_algorithms[5]], [0], [5])
    prepped_circuits = bu.quimb_setup_circuit_transform(bench_circuits)

    # For each circuit, run equivalence checking:
    for circuit in prepped_circuits:
        # Prepare data container
        data = {

        }

        # Transform to tensor networks (without initial states and cnot decomp)
        tensor_network = tnu.get_tensor_network(circuit, include_state = False, split_cnot=False)

        # Construct the plan from CoTenGra
        path = tnu.get_contraction_path(tensor_network, "cotengra")

        # Prepare gate TDDs
        gate_tdds = tddu.get_tdds_from_quimb_tensor_network(tensor_network)

        # Contract TDDs


        # Check for equivalence

        # Save data for circuit
        file_path = os.path.join(folder_path, f"circuit_{circuit}")



    # Save collected data


if __name__ == "__main__":
    first_experiment()
