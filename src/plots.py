import file_util as fu
import plotting_util as pu
import os
from enum import Enum
import math

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

def plot(folder, plots, save_path = "", inclusion_condition = (lambda file, data:True), show_3d = False):
    files = fu.load_all_json(os.path.join("experiments", folder))
    data = {v : [] for v in list(Variables)}
    file_data = {v : None for v in list(Variables)}
    for file in files:
        try:
            s, estimated_time, new_sizes = process_sizes(file)
            file_data[Variables.ESTIMATED_TIME] = ([sum(estimated_time)])
            file_data[Variables.SIZES] = (s)
            file_data[Variables.LOG_SIZES] = ([math.log10(point) for point in s])
            file_data[Variables.NEW_SIZES] = (new_sizes)
            file_data[Variables.STEPS] = (range(len(s)))
            file_data[Variables.CONTRACTION_STEPS] = (range(len(new_sizes)))
            q = file['circuit_settings']['qubits']
            file_data[Variables.NAMES] = (f"{file['circuit_settings']['algorithm']}:{q:03d}")
            file_data[Variables.QUBITS] = ([q])
            file_data[Variables.MAX_SIZES] = ([max(s)])
            file_data[Variables.LOG_MAX_SIZES] = ([math.log10(max(s))])
            file_data[Variables.CONTRACTION_TIME] = ([file["contraction_time"]])
            if "qcec_time" in file:
                file_data[Variables.QCEC_TIME] = [file["qcec_time"]]
                file_data[Variables.CIRCUIT_SETUP_TIME] = [file["circuit_setup_time"]]
                file_data[Variables.GATE_PREP_TIME] = [file["gate_prep_time"]]
                file_data[Variables.TN_CONSTRUNCTION_TIME] = [file["tn_construnction_time"]]
                file_data[Variables.PATH_CONSTRUCTION_TIME] = [file["path_construction_time"]]
            if file["path_settings"]["method"] == "cotengra":
                file_data[Variables.PATH_FLOPS] = ([math.log10(file["path_data"]["flops"])])
                file_data[Variables.PATH_SIZE] = ([math.log2(file["path_data"]["size"])])
            if "used_trials" in file["path_data"]:
                file_data[Variables.OPT_RUNS] = (range(file["path_data"]["used_trials"]))
                file_data[Variables.OPT_TIMES] = (file["path_data"]["opt_times"])
                file_data[Variables.OPT_SIZES] = ([math.log2(v) for v in file["path_data"]["opt_sizes"]])
                file_data[Variables.OPT_FLOPS] = ([math.log10(v) for v in file["path_data"]["opt_flops"]])
                file_data[Variables.OPT_WRITES] = ([math.log2(v) for v in file["path_data"]["opt_writes"]])
        except KeyError:
            ...
        except Exception as e:
            print(e)
        else: 
            if inclusion_condition(file, file_data):
                for v in list(Variables):
                    if file_data[v] is not None:
                        data[v].append(file_data[v])


    if save_path != "":
        save_path = os.path.normpath(os.path.join(os.path.realpath(__file__), "..", "..", "experiments", save_path))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    for p in plots:
        full_path = ("" if save_path == "" else os.path.join(save_path, p[-1].replace(" ", "_")))
        title = p[-1] + " " + files[0]['circuit_settings']['algorithm']
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
        elif p[0] == "3d_points": 
            if is_non_empty(data[p[1]]) and is_non_empty(data[p[2]]) and is_non_empty(data[p[3]]):
                pu.plotPoints(data[p[1]], data[p[2]], data[p[3]], [p[1].value, p[2].value, p[3].value], 
                              legend=False, series_labels=data[Variables.NAMES], marker="o", title=title, 
                              save_path="" if show_3d else full_path)
        else:
            print(f"{p[0]} is not a valid plot type!")

class Variables(Enum):
    SIZES = "Nodes |N|"
    STEPS = "Path Steps s"
    CONTRACTION_STEPS = "Path Contraction Steps s_c"
    QUBITS = "Qubits n"
    MAX_SIZES = "Max Nodes N_max"
    NAMES = "Names"
    ALGORITHM = "Algorithm"
    CONTRACTION_TIME = "Contraction Time t_c [ms]"
    ESTIMATED_TIME = "Estimated Time t_e []"
    NEW_SIZES = "Newest TDD Size N_new"
    PATH_SIZE = "Path Size log2(ps)"
    PATH_FLOPS = "Path Flops log10(pf)"
    OPT_TIMES = "Optimisation Times ot"
    OPT_WRITES = "Optimisation Writes log2(ow)"
    OPT_FLOPS = "Optimisation Flops log10(of)"
    OPT_SIZES = "Optimisation Sizes log2(os)"
    OPT_RUNS = "Optimisation Runs r"
    LOG_SIZES = "Nodes log10(|N|)"
    LOG_MAX_SIZES = "Max Nodes log10(|N_max|)"
    QCEC_TIME = "QCEC Time t_qcec [ms]"
    CIRCUIT_SETUP_TIME = "Circuit Setup Time t_cs [ms]"
    PATH_CONSTRUCTION_TIME = "Path Construction Time t_pc [ms]"
    TN_CONSTRUNCTION_TIME = "Tensor Network Construction Time t_tn [ms]"
    GATE_PREP_TIME = "Gate TDD Construction Time t_gtc [ms]"

if __name__ == "__main__":
 
    plots = [("points", Variables.QUBITS, Variables.QCEC_TIME, "QCEC Time by Qubits"),
             ("points", Variables.QUBITS, Variables.TN_CONSTRUNCTION_TIME, "Tensor Network Construction Time by Qubits"),
             ("points", Variables.QUBITS, Variables.PATH_CONSTRUCTION_TIME, "Path Construction Time by Qubits"), 
             ("points", Variables.QUBITS, Variables.GATE_PREP_TIME, "Gate TDD Construction Time by Qubits"),
             ("points", Variables.QUBITS, Variables.CIRCUIT_SETUP_TIME, "Circuit Setup Time by Qubits"),
             ("line", Variables.STEPS, Variables.SIZES, "Compulsory Sizes over Path"),
             ("line", Variables.STEPS, Variables.LOG_SIZES, "Compulsory log10 Sizes over Path"),
             ("points", Variables.QUBITS, Variables.MAX_SIZES, "Maximum Size by Qubits"),
             ("points", Variables.QUBITS, Variables.LOG_MAX_SIZES, "Maximum log10 Size by Qubits"),
             ("points", Variables.MAX_SIZES, Variables.CONTRACTION_TIME, "Time by Maximum Size"),
             ("points", Variables.ESTIMATED_TIME, Variables.CONTRACTION_TIME, "Time by Estimated Time"),
             ("points", Variables.QUBITS, Variables.ESTIMATED_TIME, "Estimated Time by Qubits"),
             ("line", Variables.CONTRACTION_STEPS, Variables.NEW_SIZES, "New Sizes over Path"),
             ("points", Variables.PATH_FLOPS, Variables.MAX_SIZES, "Max Sizes over Path Flops"),
             ("points", Variables.PATH_SIZE, Variables.MAX_SIZES, "Max Sizes over Path Size"),
             ("line", Variables.OPT_RUNS, Variables.OPT_FLOPS, "Optimisation Flops"),
             ("line", Variables.OPT_RUNS, Variables.OPT_SIZES, "Optimisation Sizes"),
             ("line", Variables.OPT_RUNS, Variables.OPT_WRITES, "Optimisation Writes"),
             ("line", Variables.OPT_RUNS, Variables.OPT_TIMES, "Optimisation Times"),
             ("points", Variables.QUBITS, Variables.CONTRACTION_TIME, "Contraction Time by Qubits"),
             ("3d_points", Variables.QUBITS, Variables.MAX_SIZES, 
                Variables.CONTRACTION_TIME, "Qubits, Maximum Size, and Contraction Time")]

    folders = [ "mapping_experiment_2023-10-19_16-48", "mapping_experiment_2023-10-19_17-08",
                "mapping_experiment_2023-10-19_17-27", "mapping_experiment_2023-10-24_09-30", 
                "mapping_experiment_2023-11-03_13-39"]
    #["driver_linear_ltr_2023-11-06_15-04"]

    #file is the raw loaded file, and data is the processed variables for that file
    inclusion_condition = lambda file, data : ("conclusive" not in file or file["conclusive"])

    for i, folder in enumerate(folders):
        plot(folder, plots, os.path.join("plots", folder), inclusion_condition=inclusion_condition, show_3d=True) 
        print(f"Plotted: {int((i + 1) / len(folders) * 100)}%")