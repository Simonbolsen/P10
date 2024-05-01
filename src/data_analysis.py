import file_util as fu
import plotting_util as pu
import os
from enum import Enum
import math
from tqdm import tqdm
import numpy as np
from contraction_plots import process_sizes
#from sklearn.linear_model import LinearRegression
from scipy import stats


def get_meta_data_from_experiment(exp_name):
    files = fu.load_all_json(os.path.join("experiments", exp_name))
    res = {}

    for file in files:
        key = file["circuit_settings"]["algorithm"]

        if key in res:
            res[key]["qubits"].append(file["circuit_settings"]["qubits"])
            for k, v in file.items():
                if not (isinstance(v, list) or isinstance(v, dict)):
                    if k in ['file_name', 'experiment_name', 'make_dataset', 'failed', 'expect_equivalence', 'version', 'folder_name']:
                        continue
                    res[key][k].append(v)
        else:
            res[key] = {}

            for k, v in file["settings"].items():
                res[key][k] = v
            for k, v in file["contraction_settings"].items():
                res[key][f"cont_{k}"] = v
            for k, v in file["path_settings"].items():
                res[key][f"path_{k}"] = v
            for k, v in file["circuit_settings"].items():
                res[key][k] = v if k != "qubits" else [v]
            for k, v in file.items():
                if not (isinstance(v, list) or isinstance(v, dict)):
                    if k in ['file_name', 'experiment_name', 'make_dataset', 'failed', 'expect_equivalence', 'version']:
                        continue
                    res[key][k] = v if k in ['folder_name'] else [v]
    
    for k, v in res.items():
        res[k]['qubits'] = sorted(list(set(v['qubits'])))
        res[k]['sub_networks'] = {key: len([v for v in res[k]['sub_networks'] if v == key]) for key in sorted(list(set(v['sub_networks'])))}
        res[k]['equivalence'] = {'True': len([v for v in res[k]['equivalence'] if v]), 'False': len([v for v in res[k]['equivalence'] if not v])}
        res[k]['conclusive'] = {'True': len([v for v in res[k]['conclusive'] if v]), 'False': len([v for v in res[k]['conclusive'] if not v])}
        res[k]["circuit_setup_time"] = sum(res[k]["circuit_setup_time"]) / len(res[k]["circuit_setup_time"])
        res[k]["tn_construnction_time"] = sum(res[k]["tn_construnction_time"]) / len(res[k]["tn_construnction_time"])

        res[k]["path_construction_time"] = sum(res[k]["path_construction_time"]) / len(res[k]["path_construction_time"])
        res[k]["contraction_time"] = sum(res[k]["contraction_time"]) / len(res[k]["contraction_time"])
        res[k]["gate_prep_time"] = sum(res[k]["gate_prep_time"]) / len(res[k]["gate_prep_time"])

    return res

    


def get_data_from_folder(folder):
    files = fu.load_all_json(os.path.join("experiments", folder))
    res = {}

    if len(files) < 1:
        return None
    method = None
    if "use_qcec_only" in files[0]["settings"] and files[0]["settings"]["use_qcec_only"]:
        method = "qcec"
    elif files[0]["path_settings"]["method"] == "linear" and files[0]["path_settings"]["use_proportional"]:
        method = "prop"
    elif files[0]["path_settings"]["method"] == "linear":
        method = "naive"
    elif files[0]["path_settings"]["method"] == "cotengra" and files[0]["path_settings"]["opt_method"] == "rgreedy":
        method = "rgreedy"
    elif files[0]["path_settings"]["method"] == "cotengra" and files[0]["path_settings"]["opt_method"] == "betweenness":
        method = "betweenness"

    is_cpp = False
    if "use_cpp_only" in files[0]["settings"] and files[0]["settings"]["use_cpp_only"]:
        is_cpp = True
    
    is_cotengra = False

    for file in tqdm(files):
        is_qcec = "use_qcec_only" in file["settings"] and file["settings"]["use_qcec_only"]
        is_cotengra = not is_qcec and (is_cotengra or file["path_settings"]["method"] == "cotengra")
        
        file_res = {}
        if not is_qcec and not is_cpp:
            s, estimated_time, new_sizes = process_sizes(file)
            file_res["max_size"] = max(s)
            file_res["avg_size"] = sum(s) / len(s)
        else:
            file_res["max_size"] = 0
            file_res["avg_size"] = 0
        file_res["algorithm"] = file["circuit_settings"]["algorithm"]
        file_res["qubits"] = file['circuit_settings']['qubits']

        file_res["cont_time"] = file["contraction_time"] if not is_qcec else file["qcec_time"]
        if "path_construction_time" in file:
            file_res["path_time"] = file["path_construction_time"]
        elif is_qcec:
            file_res["path_time"] = 0

        if file_res["cont_time"] > 5 * 60 * 1000:
            continue

        dict_key = (file_res["algorithm"], file_res["qubits"])
        if dict_key in res:
            res[dict_key].append(file_res)
        else:
            res[dict_key] = [file_res]

    return (method, is_cotengra, res)

def mean_confidence_interval(data, confidence=0.95):
    """Stolen from stack-overflow"""
    a = np.array(data, dtype=np.float32)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)

    return m, h

def get_confidence_of_data(data):
    res = {}

    for key, value in data.items():
        inter_res = {}
        cont_times = [entry["cont_time"] for entry in value]
        path_times = [entry["path_time"] for entry in value]
        max_sizes = [entry["max_size"] for entry in value]
        avg_sizes = [entry["avg_size"] for entry in value]
    
        inter_res["cont_times_conf"] = mean_confidence_interval(cont_times)
        inter_res["path_times_conf"] = mean_confidence_interval(path_times)
        inter_res["max_sizes_conf"] = mean_confidence_interval(max_sizes)
        inter_res["avg_sizes_conf"] = mean_confidence_interval(avg_sizes)
        
        res[key] = inter_res

    return res

def get_printable_conf_data(data):
    processed_data = {}
    for key, value in data.items():
        inter_res = {}
        
        for inner_key, inner_value in value.items():
            is_time = "time" in inner_key
            inter_res[inner_key] = divide_and_round_2tuple(inner_value, 1 if is_time else 1, 1 if is_time else 0)

        processed_data[key] = inter_res

    return processed_data

def divide_and_round_2tuple(value, denominator=1, decimals=1):
    if decimals == 0:
        return (round(value[0]/denominator), round(value[1]/denominator))
    return (round(value[0]/denominator, decimals), round(value[1]/denominator, decimals))

def print_table_compatible_confidence_of_data(data, conf_data, is_cotengra, is_qcec=False, is_last_column=False):
    
    for key, value in data.items():
        end_symbol = '\\\\' if is_last_column else '&&'
        print(f"{key}: ", end='')

        if is_qcec:
            conf_value = conf_data[key]
            print(f"${conf_value['cont_times_conf'][0]} \pm {conf_value['cont_times_conf'][1]}$ {end_symbol}\n")
        elif is_cotengra:
            conf_value = conf_data[key]
            #print(f"{key}: ${conf_value['max_sizes_conf'][0]} \pm {conf_value['max_sizes_conf'][1]}$", end='') 
            print(f"${conf_value['path_times_conf'][0]} \pm {conf_value['path_times_conf'][1]}$", end='')
            print(f" & ${conf_value['cont_times_conf'][0]} \pm {conf_value['cont_times_conf'][1]}$ {end_symbol}\n")
        else:
            conf_value = conf_data[key]
            #print(f"{key}: ${conf_value['max_sizes_conf'][0]} \pm {conf_value['max_sizes_conf'][1]}$", end='') 
            print(f"${conf_value['cont_times_conf'][0]} \pm {conf_value['cont_times_conf'][1]}$ {end_symbol}\n")

def print_sizes_table(data, conf_data, is_last_column=False):
    
    for key, value in data.items():
        end_symbol = '\\\\' if is_last_column else '&&'
        print(f"{key}: ", end='')

        conf_value = conf_data[key]
        #print(f"{key}: ${conf_value['max_sizes_conf'][0]} \pm {conf_value['max_sizes_conf'][1]}$", end='') 
        print(f"${conf_value['avg_sizes_conf'][0]} \pm {conf_value['avg_sizes_conf'][1]}$", end='')
        print(f" & ${conf_value['max_sizes_conf'][0]} \pm {conf_value['max_sizes_conf'][1]}$ {end_symbol}\n")
        
def print_count_table(data, is_last_column=False):
    for key, value in data.items():
        end_symbol = '\\\\' if is_last_column else '&&'
        print(f"{key}: ", end='')

        #print(f"{key}: ${conf_value['max_sizes_conf'][0]} \pm {conf_value['max_sizes_conf'][1]}$", end='') 
        print(f"{len(value)} {end_symbol}\n")

def get_all_data(folders):
    all_data = {}

    for folder in folders:
        method, _, data = get_data_from_folder(folder)
        conf_data = get_confidence_of_data(data)
        if method in all_data:
            for key, value in conf_data.items():
                all_data[method][key] = value
        else:
            all_data[method] = conf_data

    return all_data

def get_representative_circuits(all_data):
    circuits = {}
    representative_circuit_keys = []

    for method, data in all_data.items():
        inter_circs = []
        for key, value in data.items():
            if not key[0] in circuits:
                circuits[key[0]] = [key[1]]
            elif not key[1] in circuits[key[0]]:
                circuits[key[0]].append(key[1])

    for name, qubits in circuits.items():
        qubits = sorted(qubits)
        representative_circuit_keys.append((name, qubits[1]))


    return representative_circuit_keys

def prepare_data_for_bar_plot(all_data, whitelist=None, as_log=False):
    circuits = get_representative_circuits(all_data)
    method_ordering = ["qcec", "naive", "prop", "rgreedy", "betweenness"]

    if whitelist is not None:
        circuits = [(name, qubits) for (name, qubits) in circuits if name in whitelist]
                
    # for key in circuits:
    #     for method, data in all_data.items():
    #         if not key in data:
    #             data[key] = None
    prepped_data = {key: [-1 for _ in range(len(all_data.keys()))] for key in circuits}
    
    for method, data in all_data.items():
        for key, circ_data in data.items():
            if key in prepped_data:
                if circ_data is not None:
                    key_index = method_ordering.index(method)
                    prepped_data[key][key_index] = circ_data["cont_times_conf"][0] + circ_data["path_times_conf"][0]
   

    final_res = [None for _ in circuits]

    for key, value in prepped_data.items():
        final_res[whitelist.index(key[0])] = [[math.log10(v) if as_log and v > 0 else v for v in value]]

    ordered_circuits = ["" for _ in circuits]
    for algo, qubit in circuits:
        ordered_circuits[whitelist.index(algo)] = (algo, qubit)


    return final_res, ordered_circuits, method_ordering

def plot_as_bar_plot(data, groups, methods):
    prettyfied_methods = {
        "rgreedy": "RGreedy",
        "betweenness": "Betweenness",
        "naive": "Naive",
        "prop": "Proportional",
        "qcec": "QCEC"
    }

    prettyfied_algos = {
        "dj": lambda x: f"DJ\n(n={x})",
        "graphstate": lambda x: f"Graphstate\n(n={x})",
        "wstate": lambda x: f"W-State\n(n={x})",
        "twolocalrandom": lambda x: f"TwoLocalRandom\n(n={x})",
        "qpeexact": lambda x: f"QPE Exact\n(n={x})"
    }

    pretty_methods = [prettyfied_methods[method] for method in methods]
    pretty_algos = [prettyfied_algos[algo](qubit) for algo, qubit in groups]

    pu.plot_nested_bars(data, pretty_algos, pretty_methods, x_label="", y_label="Time spent $log_{10}(t_s)$ [ms]", title="Time Spent for all methods on selected algorithms")

if __name__ == "__main__":
    folders = [
        # Naive:
        "time_vanilla_naive_remaining_circs",
        # Prop:
        "time_vanilla_prop_remaining_circs",
        "time_vanilla_prop_gs_dj_ghz_64_128_256_2023-12-12_16-08",
        # QCEC:
        "time_vanilla_qcec_remaining_circs",
        "time_qcec_gs_dj_ghz_2023-12-13_11-22",
        # RGreedy
        "time_vanilla_rgreedy_remaining_circs",
        "time_vanilla_rgreedy_gs_dj_ghz_64_128_256_2023-12-11_08-30",
        # Between:
        "time_vanilla_between_remaining_circs",
        "time_sn_between_gs_dj_ghz_64_128_256_2023-12-11_11-25"
    ]

    folders = [
        "test_new_cpp_opt_benchmark_test_wsplit_betweenness_v1_"
    ]

    # all_data = get_all_data(folders)
    # prepped_data, groups, methods = prepare_data_for_bar_plot(all_data, whitelist=["dj", "graphstate", "wstate", "twolocalrandom", "qpeexact"], as_log=True)
    # plot_as_bar_plot(prepped_data, groups, methods)

    
    _, is_cotengra, data = get_data_from_folder("python_benchmark_new_")
    processed_data = get_confidence_of_data(data)
    printable_data = get_printable_conf_data(processed_data)
    print_table_compatible_confidence_of_data(data, printable_data, is_cotengra, is_qcec=False, is_last_column=False)
    # print_sizes_table(data, printable_data, is_last_column=False)
    # print_count_table(data, is_last_column=False)