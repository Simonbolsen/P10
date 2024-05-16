
from contraction_experiments import run_experiment


qb_per_alg_rep = {
    "dj": [64, 128, 256],
    "ghz": [64, 128, 256],
    "graphstate": [64, 128, 256],
    "wstate": [8, 12, 16],
    "qpeexact": [6, 8, 10],
    "qftentangled": [6, 8, 10],
    "random_eqv": [6, 8, 10],
    "realamprandom": [4, 5, 6],
    "twolocalrandom": [4, 5, 6],
    "su2random": [4, 5, 6]
}

qb_per_alg_scale = {
    "dj": [v for v in list(range(6,257,10)) if not v in []],
    "ghz": list(range(6,257,10)),
    "graphstate": [v for v in list(range(6,257,10)) if not v in []],
    "wstate": list(range(3, 17, 1)),
    "qpeexact": list(range(3, 11, 1)),
    "qftentangled": [v for v in list(range(3,11,1)) if not v in []],
    "random_eqv": list(range(3, 11, 1)),
    "realamprandom": list(range(3, 7, 1)),
    "twolocalrandom": list(range(3, 7, 1)),
    "su2random": list(range(3, 7, 1))
}


all_algs = ["ghz", "dj", "graphstate", "wstate", "qftentangled", "random_eqv", "realamprandom"]

if __name__ == "__main__":
    configs = [{
        "contraction_settings": {
            "max_time": 300, # in seconds, -1 for inf
            "max_replans": 1,
            "max_intermediate_node_size": -1 #-1 for inf
        },
        "settings": {
            "simulate": False,
            "sliced": False,
            "cnot_split": True,
            "use_subnets": True,
            "find_counter": False,
            "use_qcec_only": False,
            "use_cpp_only": [True, False, True, True, True, True, True, True, True, True][j],
            "repeat_precision": True
        },
        "path_settings": {
            "method": ["cotengra", "cotengra", "linear", "cotengra", "cpp-nngreedy", "cpp-nngreedy", "cpp-nngreedy", "cpp-lookahead", "cpp-nngreedy", "cpp-nngreedy"][j],
            "weight_function":"wf1",
            "model_name": ["n/a", "n/a", "n/a", "n/a", "model_0_jit", "model_0_jit", "biased_model_3_1_jit", "n/a", "model_d6_lr9_jit", "model_0_jit"][j],
            "opt_method": ["betweenness", "betweenness", "linear", "random-greedy", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"][j], #  kahypar-balanced, kahypar-agglom, labels, labels-agglom
            "minimize": "flops",
            "max_repeats": [1, 1, 1, 60, 1, 1, 1, 1, 1, 1][j],
            "max_time": 60,
            "use_proportional": False,
            "gridded": False,
            "linear_fraction": 0,
            "window_size": [1, 1, 1, 1, 1, 4, 1, 1, 1, 100000][j],
            "parallel": True
        },
        "circuit_settings": {
            "algorithm": alg, #"dj", "ghz", "graphstate", "qftentangled", "su2random", "twolocalrandom", "qpeexact", "wstate", "realamprandom"
            "level": (0, 2),
            "qubits": qb_per_alg_scale[alg][qb_i],
            "random_gate_deletions": 0,
            "repetition": i
        },
        "folder_name":["benchmark_scaling_w_split_betweenness", "benchmark_py_scaling_w_split_betweenness", 
                       "benchmark_scaling_new_linear", "benchmark_scaling_w_split_rgreedy", 
                       "benchmark_scaling_new_cpp_nngreedy", "benchmark_scaling_new_cpp_nngreedy_w4", 
                       "benchmark_scaling_new_cpp_biased_nngreedy", "benchmark_scaling_lookahead",
                       "benchmark_scaling_new_cpp_relaxed_nngreedy", "benchmark_scaling_offline_nngreedy"][j], #garbage
        "file_name": "standard_name",
    } for j in [0] for alg in all_algs if not alg in ["random_eqv"] for qb_i in range(len(qb_per_alg_scale[alg])) for i in range(1)]


    run_experiment(configs, folder_with_time=False, prev_rep=0)