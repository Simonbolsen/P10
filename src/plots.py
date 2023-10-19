import file_util as fu
import plotting_util as pu

def get_compulsory_sizes(data):
    sizes = data["sizes"]
    path = data["path"]

    def get_current_size(i):
        return sizes[str(i)][version_indeces[i]]
    
    def advance(i):
        version_indeces[i] += 1

    compulsory_sizes = []
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
            compulsory_sizes.append((compulsory_sizes[-1] + size) if len(compulsory_sizes) > 0 else size)

        size = compulsory_sizes[-1] - get_current_size(step[0]) - get_current_size(step[1])
        advance(step[1])
        compulsory_sizes.append(size + get_current_size(step[1]))

    return compulsory_sizes

def get_nested(ls):
    return [[v] for v in ls]

def plot(folder, save_path = ""):
    files = fu.load_all_json("experiments/" + folder)
    sizes = []
    steps = []
    names = []
    qubits = []
    max_sizes = []
    for file in files:
        s = get_compulsory_sizes(file)
        sizes.append(s)
        steps.append(range(len(s)))
        q = file['circuit_settings']['qubits']
        names.append(f"{file['circuit_settings']['algorithm']}:{q:03d}")

        qubits.append(q)
        max_sizes.append(max(s))

    pu.plot_line_series_2d(steps, sizes, names, "Steps s", "Nodes |N|", 
                           title="Compulsory Sizes", save_path="experiments/" + save_path + "/Compulsory_Sizes", legend=False)
    pu.plotPoints2d(get_nested(qubits), get_nested(max_sizes), "Qubits n", "Max Nodes N_max", 
                    legend= False, series_labels=names, title= "Maximum Nodes by Qubits", save_path="experiments/" + save_path + "/Maximum_Nodes_by_Qubits")



if __name__ == "__main__":
   plot("first_experiment_2023-10-18", "first_experiment_2023-10-18/plots") 