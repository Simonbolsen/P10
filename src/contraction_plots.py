import file_util as fu
import plotting_util as pu
import os
from enum import Enum
import math
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression

def process_sizes(data):
    sizes = data["sizes"]
    path = data["path"]

    def get_current_size(i):
        return sizes[str(i)][version_indeces[i]]
    
    def advance(i):
        version_indeces[i] += 1

    new_sizes = []
    compulsory_sizes = []
    estimated_time = []
    version_indeces = {int(i) : 0 for i in sizes.keys()}
    for step in path:
        size = 0
        flag = False

        if version_indeces[step[0]] == 0:
            advance(step[0])
            size += get_current_size(step[0])
            flag = True 
        if version_indeces[step[1]] == 0:
            advance(step[1])
            size += get_current_size(step[1])
            flag = True  
        if flag:
            estimated_time.append(size)
            compulsory_sizes.append((compulsory_sizes[-1] + size) if len(compulsory_sizes) > 0 else size)

        size = compulsory_sizes[-1] - get_current_size(step[0]) - get_current_size(step[1])
        estimated_time.append(get_current_size(step[0])**2 + get_current_size(step[1])**2)
        advance(step[1])
        compulsory_sizes.append(size + get_current_size(step[1]))
        new_sizes.append(get_current_size(step[1]))

    return compulsory_sizes, estimated_time, new_sizes

def get_nested(ls):
    return [[v] for v in ls]

def is_non_empty(l):
    return len(l) > 0

def list_extend_zeroes(l):
    max_length = max([len(l[i]) for i in range(len(l))])
    return [[k[i] if len(k) > i else 0 for i in range(max_length)] for k in l]

def find_max_inner_list(l):
    l = list_extend_zeroes(l)
    return [max([l[k][i] for k in range(len(l))]) for i in range(max([len(l[i]) for i in range(len(l))]))]

def find_sum_inner_list(l):
    l = list_extend_zeroes(l)
    return [sum([l[k][i] for k in range(len(l))]) for i in range(max([len(l[i]) for i in range(len(l))]))]

def extract_data(folder, inclusion_condition = (lambda file, data:True), silent = False):
    files = fu.load_all_json(os.path.join("experiments", folder))
    data = {v : [] for v in list(Variables)}
    file_data = {v : None for v in list(Variables)}
    for file in tqdm(files, disable=silent):
        try:
            file_data[Variables.ALGORITHM] = [file["circuit_settings"]["algorithm"]]
            q = file['circuit_settings']['qubits']
            file_data[Variables.NAMES] = (f"{file['circuit_settings']['algorithm']}:{q:03d}")
            file_data[Variables.QUBITS] = ([q])

            if "use_qcec_only" in file["settings"] and file["settings"]["use_qcec_only"]:
                file_data[Variables.QCEC_TIME] = [file["qcec_time"]]
                if inclusion_condition(file, file_data):
                    for v in list(Variables):
                        if file_data[v] is not None:
                            data[v].append(file_data[v])
                continue
            
            if "sizes" in file:
                s, estimated_time, new_sizes = process_sizes(file)
                file_data[Variables.ESTIMATED_TIME] = ([sum(estimated_time)])
                file_data[Variables.SIZES] = (s)
                file_data[Variables.LOG_SIZES] = ([math.log10(point) for point in s])
                file_data[Variables.NEW_SIZES] = (new_sizes)
                file_data[Variables.LOG2_NEW_SIZES] = ([math.log2(point) for point in new_sizes])
                file_data[Variables.STEPS] = (range(len(s)))
                file_data[Variables.CONTRACTION_STEPS] = (range(len(new_sizes)))
                file_data[Variables.MAX_SIZES] = ([max(s)])
                file_data[Variables.LOG_MAX_SIZES] = ([math.log10(max(s))])
            file_data[Variables.CONTRACTION_TIME] = ([file["contraction_time"]])
            file_data[Variables.TENSOR_COUNT] = [len(file["path"]) + (file["sub_networks"] if "sub_networks" in file else 1)]
            file_data[Variables.GATE_DELETIONS] = [file['circuit_settings']["random_gate_deletions"]]
            file_data[Variables.EQUIV_CASES] = [(1 if file["equivalence"] else 0) + 2 * (1 if file["conclusive"] else 0)]
            if "sub_networks" in file:
                file_data[Variables.SUB_NETWORK_COUNT] = [file["sub_networks"]]
            if "qcec_time" in file:
                file_data[Variables.QCEC_TIME] = [file["qcec_time"]]
            if "circuit_setup_time" in file:
                file_data[Variables.CIRCUIT_SETUP_TIME] = [file["circuit_setup_time"]]
            if "gate_prep_time" in file:
                file_data[Variables.GATE_PREP_TIME] = [file["gate_prep_time"]]
            if "tn_construnction_time" in file:
                file_data[Variables.TN_CONSTRUNCTION_TIME] = [file["tn_construnction_time"]]
            if "path_construction_time" in file:
                file_data[Variables.PATH_CONSTRUCTION_TIME] = [file["path_construction_time"]]
            if file["path_settings"]["method"] == "cotengra":
                file_data[Variables.PATH_FLOPS] = ([math.log10(file["path_data"]["flops"])])
                file_data[Variables.PATH_SIZE] = ([math.log2(file["path_data"]["size"])])
            if file["path_settings"]["method"] == "tdd_model" or (file["path_settings"]["method"] == "tree_search"):
                file_data[Variables.PREDICTED_SIZES] = file["path_data"]["size_predictions"]
            if file["path_settings"]["method"] == "tdd_model":
                file_data[Variables.MAX_PREDICTED_SIZES] = [max([p[0] for p in file["path_data"]["size_predictions"]])]
            if file["path_settings"]["method"] == "tree_search":
                file_data[Variables.MAX_PREDICTED_SIZES] = file["path_data"]["size_predictions"][file["path_data"]["chosen_sample"]]
                file_data[Variables.SIZES] = (file["path_data"]["all_size_predictions"][file["path_data"]["chosen_sample"]])
                file_data[Variables.PREDICTED_SIZE_SUM] = [sum(file_data[Variables.SIZES])]
                file_data[Variables.STEPS] = (range(len(file_data[Variables.SIZES])))
            if "alpha" in file["path_settings"]:
                file_data[Variables.ALPHA] = [file["path_settings"]["alpha"]]
            if "sample_time" in file["path_data"]:
                file_data[Variables.ALPHAS] = [file["path_settings"]["alpha"] for _ in file["path_data"]["sample_time"]]
                file_data[Variables.MAX_SAMPLE_TIME]  = [max(file["path_data"]["sample_time"]) * 1000]
                file_data[Variables.MODEL_TIME]       = [sum(t) * 1000 for t in file["path_data"]["model_time"]]
                file_data[Variables.MAX_PROPAGATION_TIME] = [max(file["path_data"]["propagation_time"]) * 1000]
                file_data[Variables.CHOICE_TIME]      = [sum(t) * 1000 for t in file["path_data"]["choice_time"]]
                file_data[Variables.STEP_TIME]        = [sum(t) * 1000 for t in file["path_data"]["step_time"]]
                file_data[Variables.INPUT_TIME]       = [sum(t) * 1000 for t in file["path_data"]["input_time"]]
                file_data[Variables.PREDICTION_TIME]  = [sum(t) * 1000 for t in file["path_data"]["prediction_time"]]
                file_data[Variables.MAX_TENSOR_TIME]      = [max(file["path_data"]["tensor_time"]) * 1000]
                file_data[Variables.MAX_EDGE_TIME]        = [max(file["path_data"]["edge_time"]) * 1000]
                file_data[Variables.ITEM_TIME]        = [sum(t) * 1000 for t in file["path_data"]["item_time"]]
                file_data[Variables.STACK_TIME]       = [sum(t) * 1000 for t in file["path_data"]["stack_time"]]
            if "version" in file and file["version"] == 1 and "used_trials" in file["path_data"]:
                file_data[Variables.OPT_RUNS_MAX] = (range(max(file["path_data"]["used_trials"])))
                file_data[Variables.OPT_TIMES_MAX] = find_max_inner_list(file["path_data"]["opt_times"])
                file_data[Variables.OPT_SIZES_MAX] = ([math.log2(v) for v in find_max_inner_list(file["path_data"]["opt_sizes"])])
                file_data[Variables.OPT_FLOPS_MAX] = ([math.log10(v) for v in find_max_inner_list(file["path_data"]["opt_flops"])])
                file_data[Variables.OPT_WRITES_MAX] = ([math.log2(v) for v in find_max_inner_list(file["path_data"]["opt_writes"])])
                file_data[Variables.OPT_RUNS_SUM] = (range(sum(file["path_data"]["used_trials"])))
                file_data[Variables.OPT_TIMES_SUM] = find_sum_inner_list(file["path_data"]["opt_times"])
                file_data[Variables.OPT_SIZES_SUM] = ([math.log2(v) for v in find_sum_inner_list(file["path_data"]["opt_sizes"])])
                file_data[Variables.OPT_FLOPS_SUM] = ([math.log10(v) for v in find_sum_inner_list(file["path_data"]["opt_flops"])])
                file_data[Variables.OPT_WRITES_SUM] = ([math.log2(v) for v in find_sum_inner_list(file["path_data"]["opt_writes"])])
            elif "used_trials" in file["path_data"]:
                file_data[Variables.OPT_RUNS] = (range(file["path_data"]["used_trials"]))
                file_data[Variables.OPT_TIMES] = (file["path_data"]["opt_times"])
                file_data[Variables.OPT_SIZES] = ([math.log2(v) for v in file["path_data"]["opt_sizes"]])
                file_data[Variables.OPT_FLOPS] = ([math.log10(v) for v in file["path_data"]["opt_flops"]])
                file_data[Variables.OPT_WRITES] = ([math.log2(v) for v in file["path_data"]["opt_writes"]])
        except KeyError as ke:
            print(ke)
        except Exception as e:
            print(e)
        else: 
            if inclusion_condition(file, file_data):
                for v in list(Variables):
                    if file_data[v] is not None:
                        data[v].append(file_data[v])

    equiv_cases = data[Variables.EQUIV_CASES]
    data[Variables.GROUP_NAMES] = {0: "Inequivalent+Inconclusive", 1: "Equivalent+Inconclusive", 2: "Inequivalent+Conclusive", 3: "Equivalent+Conclusive"}
    data[Variables.EQUIV_GROUP_COUNTS] = {data[Variables.GROUP_NAMES][i]:[[equiv_cases.count([i])]] for i in data[Variables.GROUP_NAMES].keys()}

    def smooth(i, n):
        within = 0
        s = 0
        for ii in range(i-n, i+n):
            if ii >= 0 and ii < len(equiv_cases):
                within += 1
                s += equiv_cases[ii][0]
        return s / within if within > 0 else 0

    data[Variables.SMOOTH_EQUIV_CASES] = [[smooth(i, 3)] for i, c in enumerate(equiv_cases)]
    data[Variables.NO_LABELS] = [""]

    qubit_cases = list(set(np.array(data[Variables.QUBITS]).flatten()))
    if len(qubit_cases) < 6:
        total_cases = {q:{e:[] for e in data[Variables.GROUP_NAMES].keys()} for q in qubit_cases}
        for i in range(len(data[Variables.CONTRACTION_TIME])):
            total_cases[data[Variables.QUBITS][i][0]][data[Variables.EQUIV_CASES][i][0]].append(data[Variables.CONTRACTION_TIME][i][0])

        for ok, ov in total_cases.items():
            for ik, iv in ov.items():
                total_cases[ok][ik] = [-1 if len(iv) == 0 else np.mean(iv)]

        time_data = {}
        for ok in sorted(list(total_cases.keys())):
            imd_array = np.zeros(len(data[Variables.GROUP_NAMES]))
            for ik, iv in total_cases[ok].items():
                imd_array[ik] = iv[0]
            time_data[ok] = [imd_array]

        data[Variables.CONTRACTION_TIME_BUCKETED] = time_data
        data[Variables.QUBITS_BUCKETED] = sorted(list(total_cases.keys()))
        data[Variables.GROUP_LABELS] = list(data[Variables.GROUP_NAMES].values())

    return data

def plot(folder, plots, save_path = "", inclusion_condition = (lambda file, data:True), show_3d = False, silent = False):
    if type(folder) == str:
        data = extract_data(folder, inclusion_condition, silent)
    else:
        all_data = [extract_data(f, inclusion_condition) for f in folder]
        data = {}
        uncombinable_variables = [Variables.EQUIV_GROUP_COUNTS, Variables.GROUP_NAMES, Variables.CONTRACTION_TIME_BUCKETED, Variables.QUBITS_BUCKETED]
        for key in all_data[0]:
            if key not in uncombinable_variables:
                data[key] = [[x for l in d[key] for x in l] for d in all_data]

    if save_path != "":
        save_path = os.path.normpath(os.path.join(os.path.realpath(__file__), "..", "..", "experiments", save_path))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    for p in tqdm(plots):
        full_path = ("" if save_path == "" else os.path.join(save_path, p[-1].replace(" ", "_")))
        title = p[-1] + " " + data[Variables.ALGORITHM][0][0]
        if p[0] == "line":
            if is_non_empty(data[p[1]]) and is_non_empty(data[p[2]]):
                pu.plot_line_series_2d(data[p[1]], data[p[2]], data[Variables.NAMES], 
                                        p[1].value, p[2].value, title=title, 
                                        save_path=full_path, legend=False)
        elif p[0] == "points": 
            if is_non_empty(data[p[1]]) and is_non_empty(data[p[2]]):
                pu.plotPoints2d(data[p[1]], data[p[2]], p[1].value, p[2].value, 
                                series_labels=data[Variables.NAMES], title= title,
                                marker="o", save_path=full_path, legend=False)
            else:
                print(f"{p[1].value}: {is_non_empty(data[p[1]])}, {p[2].value}: {is_non_empty(data[p[2]])}")
        elif p[0] == "3d_points": 
            if is_non_empty(data[p[1]]) and is_non_empty(data[p[2]]) and is_non_empty(data[p[3]]):
                pu.plotPoints(data[p[1]], data[p[2]], data[p[3]], [p[1].value, p[2].value, p[3].value], 
                              legend=False, series_labels=data[Variables.NAMES], marker="o", title=title, 
                              save_path="" if show_3d else full_path)
        elif p[0] == "bar":
            if is_non_empty(data[p[1]]) and is_non_empty(data[p[2]]):
                values = list(data[p[2]].values())
                groups = list(data[p[2]].keys())
                pu.plot_nested_bars(values, groups, data[p[3]], x_label=p[1].value, y_label=p[2].value, title=p[4], save_path=full_path)
        else:
            print(f"{p[0]} is not a valid plot type!")

def gate_del_comparison_plots(save_path = "", inclusion_condition = (lambda file, data:True)): 
    folders =  [ 
        "gate_deletion_mapping_ghz_256_2023-12-08_10-25",
        "gate_deletion_mapping_dj_256_2023-12-08_10-59"]
    
    data = [extract_data(folder, inclusion_condition) for folder in folders]
    variables = [Variables.CONTRACTION_TIME, Variables.GATE_DELETIONS, Variables.MAX_SIZES]
    plot_data = {v:[[d[0] for d in experiment[v]] for experiment in data] for v in variables}
    names = [experiment[Variables.ALGORITHM][0][0] for experiment in data]

    variables = {Variables.CONTRACTION_TIME_SLOPE : Variables.CONTRACTION_TIME, Variables.MAX_SIZE_SLOPE : Variables.MAX_SIZES}
    plot_data |= {k:[[d[0]/(1 + experiment[Variables.GATE_DELETIONS][i][0]) for i, d in enumerate(experiment[v])]for experiment in data]  for k, v in variables.items()}

    if save_path != "":
        save_path = os.path.normpath(os.path.join(os.path.realpath(__file__), "..", "..", "experiments", save_path))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    #gate_del_linear_analysis(plot_data, save_path, names)
    
    plots = [("comparison", Variables.GATE_DELETIONS, Variables.CONTRACTION_TIME_SLOPE, "Contraction Time Slope"),
             ("comparison", Variables.GATE_DELETIONS, Variables.MAX_SIZE_SLOPE, "Maximum TDD size Slope")]
    
    plots = [("comparison", Variables.GATE_DELETIONS, Variables.CONTRACTION_TIME, "Contraction Time"),
             ("comparison", Variables.GATE_DELETIONS, Variables.MAX_SIZES, "Maximum TDD size")]
    for p in plots:
        title = p[3] + f" by Gate Deletion"
        full_path = ("" if save_path == "" else os.path.join(save_path, (p[3] + f" GD").replace(" ", "_")))
        pu.plotPoints2d(plot_data[p[1]], plot_data[p[2]], p[1].value, p[2].value, 
                                    series_labels=names, title= title,
                                        marker="o", marker_size=10, save_path=full_path, legend=True)

def gate_del_linear_analysis(plot_data, save_path, names) :
    dj_linear_points = []
    dj_linear_qubits = []
    dj_quadratic_points = []
    dj_quadratic_qubits = []

    for s, series in enumerate(plot_data[Variables.CONTRACTION_TIME]):

        dj_linear_points.append([])
        dj_linear_qubits.append([])
        dj_quadratic_points.append([])
        dj_quadratic_qubits.append([])

        for p, point in enumerate(series):
            n = plot_data[Variables.GATE_DELETIONS][s][p]
            algorithm = names[s]
            if algorithm == "ghz":
                split = 900 + 30 * n
            else:
                raise Exception(f"Unkown algorithm: {algorithm}")

            if point > split:
                dj_quadratic_points[s].append(plot_data[Variables.CONTRACTION_TIME][s][p])
                dj_quadratic_qubits[s].append(n)
            else:
                dj_linear_points[s].append(plot_data[Variables.CONTRACTION_TIME][s][p])
                dj_linear_qubits[s].append(n)
    
    labels = [f"{n}, {t}" for t in ["low", "high"] for n in names]
    q = dj_linear_qubits + dj_quadratic_qubits
    v = dj_linear_points + dj_quadratic_points
    trends = []

    for i, n in enumerate(q):
        x = np.array(n)
        y = np.array(v[i])
        coefficients = np.polyfit(x, y, 1)
        trend_line = np.poly1d(coefficients)
        predicted_y = trend_line(x)
        residuals = y - predicted_y
        ss_residual = np.sum(residuals**2)
        ss_total = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_residual / ss_total)

        print(f"{labels[i]}: {print_polynomial(coefficients, 2)}, r^2: {round_sig(r_squared, 2)}")
        trends.append(coefficients)

    title = f"Contraction Time by Gate Deletion"
    full_path = ("" if save_path == "" else os.path.join(save_path, (f"Contraction Time Clustered GD").replace(" ", "_")))
    pu.plotPoints2d(q, v, Variables.QUBITS.value, Variables.CONTRACTION_TIME.value, trends = trends,
                                        series_labels=labels , title= title,
                                        marker="o", marker_size=10, save_path=full_path, legend=True)

def qubit_comparison_plots(save_path = "", inclusion_condition = (lambda file, data:True)): 
    folders =  [ 
        "equivalent_ghz_rgreedy_2023-12-07_08-18",
        "inequivalent_ghz_del_1_2023-12-06_20-56",
        "inequivalent_ghz_del_3_2023-12-06_21-12",

        "driver_rgreedy_2023-11-10_10-06",
        "inequivalent_gate_del_1_2023-11-14_09-31",
        "inequivalent_gate_del_3_2023-11-14_10-48",

        "sub_network_effect_without_2023-11-13_18-49",
        "inequivalent_graph_del_1_2023-11-14_12-17",
        "inequivalent_graph_del_3_2023-11-14_18-29",]
    
    data = [extract_data(folder, inclusion_condition) for folder in folders]

    variables = [Variables.CONTRACTION_TIME, Variables.QUBITS, Variables.MAX_SIZES]
    algorithms = ["ghz", "dj", "graphstate"]

    plot_data = {algorithm: {v:[[d[0] for d in experiment[v]] for experiment in data if experiment[Variables.ALGORITHM][0][0] == algorithm] for v in variables} for algorithm in algorithms}

    names = {algorithm:[f"Gate Deletion: {experiment[Variables.GATE_DELETIONS][0][0]}" for experiment in data if experiment[Variables.ALGORITHM][0][0] == algorithm] for algorithm in algorithms}

    variables = {Variables.CONTRACTION_TIME_SLOPE : Variables.CONTRACTION_TIME, Variables.MAX_SIZE_SLOPE : Variables.MAX_SIZES}
    for algorithm in algorithms:
        plot_data[algorithm] |= {k:[[d[0]/experiment[Variables.QUBITS][i][0] for i, d in enumerate(experiment[v])] 
                                for experiment in data if experiment[Variables.ALGORITHM][0][0] == algorithm] for k, v in variables.items()}

    if save_path != "":
        save_path = os.path.normpath(os.path.join(os.path.realpath(__file__), "..", "..", "experiments", save_path))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    for algorithm in algorithms:
        linear_analysis(plot_data, save_path, names, algorithm)
    
    plots = [("comparison", Variables.QUBITS, Variables.CONTRACTION_TIME_SLOPE, "Contraction Time Slope"),
             ("comparison", Variables.QUBITS, Variables.MAX_SIZE_SLOPE, "Maximum TDD size Slope")]

    plots = [("comparison", Variables.QUBITS, Variables.CONTRACTION_TIME, "Contraction Time"),
             ("comparison", Variables.QUBITS, Variables.MAX_SIZES, "Maximum TDD size")]
    for algorithm in algorithms:
        for p in plots:
            title = p[3] + f" on the {algorithm} algorithm"
            full_path = ("" if save_path == "" else os.path.join(save_path, (p[3] + f" {algorithm}").replace(" ", "_")))
            pu.plotPoints2d(plot_data[algorithm][p[1]], plot_data[algorithm][p[2]], p[1].value, p[2].value, 
                                        series_labels=names[algorithm], title= title,
                                        marker="o", marker_size=10, save_path=full_path, legend=True)

def round_sig(number, fig):
    if number == 0:
        return 0  # Handling special case for 0

    return round(number, -int(math.floor(math.log10(abs(number)))) + fig - 1)

def print_polynomial(coefficients, sig_fig):
    degree = len(coefficients) - 1
    poly_string = ""

    for i, coeff in enumerate(coefficients):
        power = degree - i

        if coeff != 0:
            c = round_sig(coeff, sig_fig)
            if power == 0:
                poly_string += f"{c}"
            elif power == 1:
                poly_string += f"{c}*x"
            else:
                poly_string += f"{c}*x^{power}"

            if i < len(coefficients) - 1:
                poly_string += " + "

    return poly_string

def linear_analysis(plot_data, save_path, names, algorithm = "dj") :
    dj_linear_points = []
    dj_linear_qubits = []
    dj_quadratic_points = []
    dj_quadratic_qubits = []

    for s, series in enumerate(plot_data[algorithm][Variables.CONTRACTION_TIME_SLOPE]):

        dj_linear_points.append([])
        dj_linear_qubits.append([])
        dj_quadratic_points.append([])
        dj_quadratic_qubits.append([])

        for p, point in enumerate(series):
            n = plot_data[algorithm][Variables.QUBITS][s][p]

            if algorithm == "dj":
                split = 10
            elif algorithm == "graphstate":
                split = 10 + 15 / 225.0 * n
            elif algorithm == "ghz":
                split = 6.5
            else:
                raise Exception(f"Unkown algorithm: {algorithm}")

            if point > split:
                dj_quadratic_points[s].append(plot_data[algorithm][Variables.CONTRACTION_TIME][s][p])
                dj_quadratic_qubits[s].append(n)
            else:
                dj_linear_points[s].append(plot_data[algorithm][Variables.CONTRACTION_TIME][s][p])
                dj_linear_qubits[s].append(n)
    
    labels = [f"{n}, {t}" for t in ["linear", "quadratic"] for n in names[algorithm]]
    q = dj_linear_qubits + dj_quadratic_qubits
    v = dj_linear_points + dj_quadratic_points
    trends = []

    for i, n in enumerate(q):
        x = np.array(n)
        y = np.array(v[i])
        coefficients = np.polyfit(x, y, 1 if "linear" in labels[i] else 2)
        trend_line = np.poly1d(coefficients)
        predicted_y = trend_line(x)
        residuals = y - predicted_y
        ss_residual = np.sum(residuals**2)
        ss_total = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_residual / ss_total)

        print(f"{labels[i]}: {print_polynomial(coefficients, 2)}, r^2: {round_sig(r_squared, 2)}")
        trends.append(coefficients)

    title = f"Contraction Time on the {algorithm} algorithm"
    full_path = ("" if save_path == "" else os.path.join(save_path, (f"Contraction Time Clustered {algorithm}").replace(" ", "_")))
    pu.plotPoints2d(q, v, Variables.QUBITS.value, Variables.CONTRACTION_TIME.value, trends = trends,
                                        series_labels=labels , title= title,
                                        marker="o", marker_size=10, save_path=full_path, legend=True)

class Variables(Enum):
    SIZES = "Nodes |N|"
    STEPS = "Path Steps s"
    CONTRACTION_STEPS = "Path Contraction Steps s_c"
    QUBITS = "Qubits n"
    MAX_SIZES = "Max Nodes $|N_{max}|$"
    NAMES = "Names"
    ALGORITHM = "Algorithm"
    CONTRACTION_TIME = "Contraction Time $t_c$ [ms]"
    ESTIMATED_TIME = "Estimated Time t_e []"
    NEW_SIZES = "Newest TDD Size N_new"
    PATH_SIZE = "Path Size log2(ps)"
    PATH_FLOPS = "Path Flops log10(pf)"
    OPT_TIMES = "Optimisation Times ot"
    OPT_WRITES = "Optimisation Writes log2(ow)"
    OPT_FLOPS = "Optimisation Flops log10(of)"
    OPT_SIZES = "Optimisation Sizes log2(os)"
    OPT_RUNS = "Optimisation Runs r"
    OPT_TIMES_MAX = "Max Optimisation Times ot"
    OPT_WRITES_MAX = " Max Optimisation Writes log2(ow)"
    OPT_FLOPS_MAX = "Max Optimisation Flops log10(of)"
    OPT_SIZES_MAX = "Max Optimisation Sizes log2(os)"
    OPT_RUNS_MAX = "Max Optimisation Runs r"
    OPT_TIMES_SUM = "Sum of Optimisation Times ot"
    OPT_WRITES_SUM = "Sum of Optimisation Writes log2(ow)"
    OPT_FLOPS_SUM = "Sum of Optimisation Flops log10(of)"
    OPT_SIZES_SUM = "Sum of Optimisation Sizes log2(os)"
    OPT_RUNS_SUM = "Sum of Optimisation Runs r"
    LOG_SIZES = "Nodes log10(|N|)"
    LOG2_NEW_SIZES = "Nodes log2(|N_new|)"
    LOG_MAX_SIZES = "Max Nodes log10(|N_max|)"
    QCEC_TIME = "QCEC Time t_qcec [ms]"
    CIRCUIT_SETUP_TIME = "Circuit Setup Time t_cs [ms]"
    PATH_CONSTRUCTION_TIME = "Path Construction Time t_pc [ms]"
    TN_CONSTRUNCTION_TIME = "Tensor Network Construction Time t_tn [ms]"
    GATE_PREP_TIME = "Gate TDD Construction Time t_gtc [ms]"
    SUB_NETWORK_COUNT = "Num of Sub Networks"
    TENSOR_COUNT = "Num of Tensors"
    GATE_DELETIONS = "Num of Gates Deletions"
    GROUP_NAMES = "Names for Equivalence Cases"
    EQUIV_GROUP_COUNTS = "Group Count"
    EQUIV_CASES = "Equivalence Cases"
    SMOOTH_EQUIV_CASES = "Smoothed Equivalence Cases"
    CONTRACTION_TIME_BUCKETED = "Contraction Time in Buckets t_c [ms]"
    QUBITS_BUCKETED = "Qubits in Buckets n"
    NO_LABELS = "No Labels"
    GROUP_LABELS = "Group Names as Labels"
    CONTRACTION_TIME_SLOPE = "Contraction time slope"
    MAX_SIZE_SLOPE = "Max size slope"
    PREDICTED_SIZES = "Predicted Sizes"
    MAX_PREDICTED_SIZES = "Max Predicted Sizes"
    PREDICTED_SIZE_SUM = "Sum of Predicted Sizes"
    ALPHA = "Alpha value"
    ALPHAS = "Alpha value "
    MAX_SAMPLE_TIME = "Maximum Sample Time [ms]"
    MODEL_TIME = "Model Time [ms]"
    SAMPLE_TIME = "Sample Time [ms]"
    MAX_PROPAGATION_TIME = "Maximum Propagation Time [ms]"
    CHOICE_TIME = "Choice Time [ms]" 
    STEP_TIME = "Step Time [ms]"
    INPUT_TIME = "Input Time [ms]"
    PREDICTION_TIME = "Prediction Time [ms]"
    MAX_TENSOR_TIME = "Tensor Time [ms]" 
    MAX_EDGE_TIME = "Edge Time [ms]"
    ITEM_TIME = "Item Time [ms]"
    STACK_TIME = "Stack Time [ms]"

if __name__ == "__main__":
 
    plots = [
            #  ("points", Variables.QUBITS, Variables.QCEC_TIME, "QCEC Time by Qubits"),
            #  ("points", Variables.QUBITS, Variables.TN_CONSTRUNCTION_TIME, "Tensor Network Construction Time by Qubits"),
            #  ("points", Variables.QUBITS, Variables.PATH_CONSTRUCTION_TIME, "Path Construction Time by Qubits"), 

            #  ("points", Variables.QUBITS, Variables.MAX_PREDICTED_SIZES, "Maximum Predicted Sizes by Qubits"), 
              ("points", Variables.PATH_CONSTRUCTION_TIME, Variables.MAX_PREDICTED_SIZES, "Maximum Predicted Sizes by Path Construction Time"), 
              ("points", Variables.PATH_CONSTRUCTION_TIME, Variables.CONTRACTION_TIME, "Contraction Time by Path Construction Time"), 
            #  ("points", Variables.QUBITS, Variables.GATE_PREP_TIME, "Gate TDD Construction Time by Qubits"),
            #  ("points", Variables.QUBITS, Variables.CIRCUIT_SETUP_TIME, "Circuit Setup Time by Qubits"),
            #  ("points", Variables.QUBITS, Variables.SUB_NETWORK_COUNT, "Num of Sub Networks by Qubits"),
            #  ("points", Variables.QUBITS, Variables.TENSOR_COUNT, "Num of Tensors by Qubits"),
            #  ("line", Variables.STEPS, Variables.SIZES, "Compulsory Sizes over Path"),
            #  ("line", Variables.STEPS, Variables.LOG_SIZES, "Compulsory log10 Sizes over Path"),
            #  ("points", Variables.QUBITS, Variables.MAX_SIZES, "Maximum Size by Qubits"),
            #  ("points", Variables.QUBITS, Variables.LOG_MAX_SIZES, "Maximum log10 Size by Qubits"),
            #  ("points", Variables.MAX_SIZES, Variables.CONTRACTION_TIME, "Time by Maximum Size"),
            #  ("points", Variables.ESTIMATED_TIME, Variables.CONTRACTION_TIME, "Time by Estimated Time"),
            #  ("points", Variables.QUBITS, Variables.ESTIMATED_TIME, "Estimated Time by Qubits"),
            #  ("line", Variables.CONTRACTION_STEPS, Variables.NEW_SIZES, "New Sizes over Path"),
            #  ("points", Variables.PATH_FLOPS, Variables.MAX_SIZES, "Max Sizes over Path Flops"),
            #  ("points", Variables.PATH_SIZE, Variables.MAX_SIZES, "Max Sizes over Path Size"),
            #  ("line", Variables.OPT_RUNS, Variables.OPT_FLOPS, "Optimisation Flops"),
            #  ("line", Variables.OPT_RUNS, Variables.OPT_SIZES, "Optimisation Sizes"),
            #  ("line", Variables.OPT_RUNS, Variables.OPT_WRITES, "Optimisation Writes"),
            #  ("line", Variables.OPT_RUNS, Variables.OPT_TIMES, "Optimisation Times"),  
            #  ("points", Variables.ALPHAS, Variables.MODEL_TIME, "Model Time by Alpha value"),
            #  ("points", Variables.ALPHAS, Variables.CHOICE_TIME, "Choice Time by Alpha value"),
            #  ("points", Variables.ALPHAS, Variables.STEP_TIME, "Step Time by Alpha value"),
            #  ("points", Variables.ALPHAS, Variables.INPUT_TIME, "Input Time by Alpha value"),
            #  ("points", Variables.ALPHAS, Variables.PREDICTION_TIME, "Preidction Time by Alpha value"),
            #  ("points", Variables.ALPHA, Variables.MAX_TENSOR_TIME, "Maximum Tensor Time by Alpha value"),
            #  ("points", Variables.ALPHA, Variables.MAX_EDGE_TIME, "Maximum Edge Time by Alpha value"),
            #  ("points", Variables.ALPHAS, Variables.ITEM_TIME, "Item Time by Alpha value"),
            #  ("points", Variables.ALPHAS, Variables.STACK_TIME, "Stack Time by Alpha value"),
              ("points", Variables.ALPHA, Variables.CONTRACTION_TIME, "Contraction Time by Alpha value"),
              ("line", Variables.STEPS, Variables.SIZES, "Predicted Sizes over Path"),
            #  ("points", Variables.QUBITS, Variables.CONTRACTION_TIME, "Contraction Time by Qubits"),
            #  ("points", Variables.ALPHA, Variables.MAX_SAMPLE_TIME, "Maximum Sample Time by Alpha value"),
            #  ("points", Variables.ALPHA, Variables.MAX_PROPAGATION_TIME, "Maximum Propagation Time by Alpha value"),
              ("points", Variables.MAX_PREDICTED_SIZES, Variables.CONTRACTION_TIME, "Contraction Time by Maximum Predicted Sizes"),
              ("points", Variables.PREDICTED_SIZE_SUM, Variables.CONTRACTION_TIME, "Contraction Time by Sum of Predicted Sizes"),
              ("points", Variables.ALPHAS, Variables.PREDICTED_SIZES, "Predicted Sizes by Alpha Value"),
            #  ("points", Variables.ALPHA, Variables.MAX_PREDICTED_SIZES, "Max Predicted Sizes by Alpha Value"),
             # ("points", Variables.MAX_PREDICTED_SIZES, Variables.CONTRACTION_TIME, "Contraction Time by Predicted Maximum Sizes"),
            #  ("points", Variables.GATE_DELETIONS, Variables.MAX_SIZES, "Max size by Gate Deletion"),
            #  ("points", Variables.GATE_DELETIONS, Variables.QCEC_TIME, "QCEC Time by Gate Deletion"),
            #  ("points", Variables.GATE_DELETIONS, Variables.TN_CONSTRUNCTION_TIME, "Tensor Network Construction Time by Gate Deletion"),
            #  ("points", Variables.GATE_DELETIONS, Variables.PATH_CONSTRUCTION_TIME, "Path Construction Time by Gate Deletion"),
            #  ("points", Variables.GATE_DELETIONS, Variables.GATE_PREP_TIME, "Gate TDD Construction Time by Gate Deletion"),
            #  ("points", Variables.GATE_DELETIONS, Variables.CIRCUIT_SETUP_TIME, "Circuit Setup Time by Gate Deletion"),
            #  ("points", Variables.GATE_DELETIONS, Variables.SUB_NETWORK_COUNT, "Num of Sub Networks by Gate Deletion"),
            #  ("points", Variables.GATE_DELETIONS, Variables.TENSOR_COUNT, "Num of Tensors by Gate Deletion"),
            #  ("points", Variables.GATE_DELETIONS, Variables.LOG_MAX_SIZES, "Maximum log10 Size by Gate Deletion"),
            #  ("points", Variables.GATE_DELETIONS, Variables.ESTIMATED_TIME, "Estimated Time by Gate Deletion"),
            #  ("points", Variables.GATE_DELETIONS, Variables.CONTRACTION_TIME, "Contraction Time by Gate Deletion"),
            #  ("points", Variables.QUBITS, Variables.EQUIV_CASES, "Equivalence Case by Qubits"),
            #  ("points", Variables.QUBITS, Variables.SMOOTH_EQUIV_CASES, "Smooth Equivalence Case by Qubits"),
            #("bar", Variables.GROUP_NAMES, Variables.EQUIV_GROUP_COUNTS, Variables.NO_LABELS, "Count of Equivalence Cases"),
            #  ("bar", Variables.QUBITS_BUCKETED, Variables.CONTRACTION_TIME_BUCKETED, Variables.GROUP_LABELS, "Contraction Time by Qubits and Equivalence Cases"),

            #  ("3d_points", Variables.QUBITS, Variables.MAX_SIZES, 
            #     Variables.CONTRACTION_TIME, "Qubits, Maximum Size, and Contraction Time")
                ]

    # ["simulation_dj_gate_del_1_2023-11-17_11-21"]
    folders = ["ts_mV_w2_wstate__"]#, ["ts_mV_w2_dj_alpha_", "data_model_V_c_un_dj_","data_model_V_cc_un_dj_", "data_tree_search_model_V_cc_un_dj_", "tree_search_model_V_dj_"]]
    
    #data = extract_data("model_contraction_2024-03-06_14-20")
    ...

    #file is the raw loaded file, and data is the processed variables for that file
    inclusion_condition = lambda file, data : ("conclusive" not in file or file["conclusive"] or file["settings"]["simulate"]) and file["contraction_time"] < 10000

    #gate_del_comparison_plots(os.path.join("plots", "comparison_plots"), inclusion_condition=inclusion_condition) 

    for i, folder in enumerate(folders):
        if type(folder) == str:
           save_path = os.path.join("plots", folder)
        else:
            save_path = os.path.join("plots/comparison_plots", "_".join(folder))

        plot(folder, plots, save_path, inclusion_condition=inclusion_condition, show_3d=True) 
        print(f"Plotted: {int((i + 1) / len(folders) * 100)}%")