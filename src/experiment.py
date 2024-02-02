from quimb.tensor import Circuit
import cotengra as ctg
import numpy as np
import random
from tddpure.TDD.TDD import Ini_TDD, TDD, tdd_2_np, cont
from tddpure.TDD.TN import Index,Tensor,TensorNetwork
from tddpure.TDD.TDD_Q import cir_2_tn,get_real_qubit_num,add_trace_line,add_inputs,add_outputs
import circuit_util as cu
import tdd_util as tddu
from tdd_util import reverse_lexicographic_key
import tensor_network_util as tnu
import bench_util as bu
import os
from datetime import datetime
import time
import json
import tn_draw
from tqdm import tqdm
from itertools import combinations
from mqt.qcec import verify
import argparse
from tabulate import tabulate
from first_experiment import first_experiment

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


import matplotlib as mpl
mpl.use("TkAgg")

argparser = argparse.ArgumentParser()

experiment_types = {
    "formal": 0,
    "simulation": 1
}

setup_types = {
    "standard": 0,
    "pure_qcec": 1
}

methods = {
    "cotengra": 0,
    "linear": 1,
    "nn_model": 2
}

path_methods = {
    "rgreedy": 0,
    "betweenness": 1,
    "walktrap": 2,
    "linear": 3,
    "proportional": 4
}

minimize_methods = {
    "flops": 0,
    "size": 1,
    "combo": 2
}

argparser.add_argument('--exp_name', dest='exp_name', type=str)
argparser.add_argument('--exp_type', dest="exp_type", type=str, default="formal", choices=experiment_types.keys())
argparser.add_argument('-debug', dest="debug", action="store_true")
argparser.add_argument('-counter', dest="find_counter", action="store_true")
argparser.add_argument('-ts', dest="use_time_stamp", action="store_true")

# --------------- Circuit Settings ---------------
argparser.add_argument('--setup', dest='setup', type=str, default="standard", choices=setup_types.keys())
argparser.add_argument('--algorithms', dest='algorithms', nargs='+', type=str, default=["dj", "ghz"])
argparser.add_argument('--levels', dest='levels', nargs='+', type=lambda a: tuple(map(int, a.split(','))), default=[(0,2)])
argparser.add_argument('--qubit_range', dest='qubit_range', type=lambda a: tuple(map(int, a.split(','))), default=None, help="Takes tuples of (start, stop, step)")
argparser.add_argument('--qubits', dest='qubits', nargs='+', type=int, default=[4,6,8], help="Takes tuples of (start, stop, step)")
argparser.add_argument('--gate_deletions', dest='num_of_gate_deletions', type=int, default=0)
argparser.add_argument('--reps', dest='repetition', type=int, default=1)
argparser.add_argument('-sliced', dest="sliced", action="store_true")
argparser.add_argument('-split', dest="cnot_split", action="store_true")
argparser.add_argument('-subnets', dest="use_subnets", action="store_true")



# --------------- Contraction Settings ---------------
argparser.add_argument('--max_cont_time', dest='max_cont_time', type=int, default=-1)
argparser.add_argument('--max_attemps', dest='max_attemps', type=int, default=1)
argparser.add_argument('--max_int_node_size', dest='max_int_node_size', type=int, default=-1)

# --------------- Path Settings ---------------
argparser.add_argument('--method', dest='method', type=str, default="cotengra", choices=methods.keys())
argparser.add_argument('--path_method', dest='path_method', type=str, default="rgreedy", choices=path_methods.keys())
argparser.add_argument('--minimize', dest='minimize', type=str, default='flops', choices=minimize_methods.keys())
argparser.add_argument('--max_repeats', dest='max_repeats', type=int, default=50)
argparser.add_argument('--max_plan_time', dest='max_plan_time', type=int, default=60)
argparser.add_argument('--linear_frac', dest='linear_fraction', type=float, default=0.0)
argparser.add_argument('--model_name', dest='model_name', type=str, default='')

# --------------- Saving Settings ---------------
argparser.add_argument('-int_res', dest="save_intermediate_results", action="store_true")
argparser.add_argument('-draw', dest="draw", action="store_true")
argparser.add_argument('-print', dest="should_print", action="store_true")


supported_algorithms = [
    "ghz",
    "graphstate",
    "twolocalrandom", 
    "qftentangled",
    "dj",
    "qpeexact", 
    "su2random",
    "wstate",
    "realamprandom"
]

def cprint(text: str):
    if should_print:
        print(text)

def legal_args(args):
    for algorithm in args.algorithms:
        if not algorithm in supported_algorithms:
            print(f"Algorithm {algorithm} not supported.")
            return False

    for (left, right) in args.levels:
        if not (left in range(4) and right in range(4)):
            print(f"Level ({left}, {right}) is not valid.")
            return False

    if args.qubit_range is not None:
        for (start, stop, step) in [args.qubit_range]:
            if start < 2 or start >= stop:
                print(f"Qubits must be <=2 and stop must be greater than start")
                return False
        
    if args.num_of_gate_deletions < 0:
        print(f"Gate deletions cannot be negative")
        return False
    
    if args.linear_fraction < 0.0 or args.linear_fraction > 1.0:
        print(f"Linear fraction must be in [0.0, 1.0]")
        return False

    return True

def pretty_print(args):
    return tabulate(vars(args).items(), headers=["arg", "value"], missingval=f"{bcolors.WARNING}None{bcolors.ENDC}")


def run_experiment(args):
    if not legal_args(args):
        raise argparse.ArgumentError("Illegal arguments")

    pretty_print(args)

    global should_print
    should_print = args.should_print


    contraction_settings = {
                "max_time": args.max_cont_time, # in seconds, -1 for inf
                "max_replans": args.max_attemps,
                "max_intermediate_node_size": args.max_int_node_size #-1 for inf
            }

    path_settings = {
                "method": args.method,
                "opt_method": args.path_method, 
                "minimize": args.minimize,
                "max_repeats": args.max_repeats,
                "max_time": args.max_plan_time,
                "use_proportional": args.method == "linear" and args.path_method == "proportional",
                "gridded": False,
                "linear_fraction": args.linear_fraction
            }

    iter_settings = {
        "algorithms": args.algorithms,
        "levels": args.levels,
        "qubits": list(range(*args.qubit_range)) if args.qubit_range is not None else args.qubits, # NOT CORRECT
        "random_gate_dels_range": [0],
        "repetitions": args.repetition
    }

    settings = {
        "simulate": args.exp_type == "simulation",
        "sliced": args.sliced,
        "cnot_split": args.cnot_split,
        "use_subnets": args.use_subnets,
        "find_counter": args.find_counter,
        "use_qcec_only": args.exp_type == "pure_qcec"
    }

    first_experiment(iter_settings=iter_settings, settings=settings, 
                     contraction_settings=contraction_settings, path_settings=path_settings,
                     folder_name=args.exp_name, folder_with_time=args.use_time_stamp)


if __name__ == "__main__":
    args = argparser.parse_args()
    run_experiment(args)
    #first_experiment()
