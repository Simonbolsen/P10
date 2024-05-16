import math
import time
import torch
from tqdm import tqdm
from random import random
from tdd_nn import get_single_value_tensor
from tdd_nn import get_tensors
from tdd_nn import get_tensor
from tdd_nn import call_model

def weight_function_1(prediction, bound, visits, path_prediction, path_bound, sample_num):
    return bound - prediction + visits * (path_bound - path_prediction)

def get_weight_function_2(alpha):
    return lambda prediction, bound, visits, path_prediction, path_bound, sample_num: alpha**(- prediction)

def get_weight_function_3(alpha, beta):
    return lambda prediction, bound, visits, path_prediction, path_bound, sample_num: alpha**(bound * beta - prediction)

def get_edges(index_sets, tensor_network):
    edges = {}

    for o in tensor_network.ind_map.values():
        key = (min(o), max(o))
        if key[0] != key[1]:
            if key in edges:
                edges[key] += 1
            else:
                edges[key] = 1
            
            index_sets[key[0]].add(key[1])
            index_sets[key[1]].add(key[0])

    return edges

def get_path(model, tensor_network, print_sizes = False, data = None, normalized = True):
    if data is not None:
        data["path_data"]["size_predictions"] = []

    path = []
    tensors, index_sets = get_tensors(tensor_network)
    edges = get_edges(index_sets, tensor_network)
    
    progress_bar = tqdm(total=len(tensors) - 1, desc="Countdown", unit="step")

    prediction_times = 0
    cleanup_time = 0

    new_edges = list(edges.keys())
    edge_predictions = {}

    while len(edges) > 0:
        progress_bar.update(1)

        prediction_times -= time.time()

        input = {"left_values":[], "right_values": [], "shared_values": []}

        for e in new_edges:
            input["left_values"].append(tensors[e[0]])
            input["right_values"].append(tensors[e[1]])
            input["shared_values"].append(get_single_value_tensor(edges[e]))

        input["left_values"] = torch.stack(input["left_values"])
        input["right_values"] = torch.stack(input["right_values"])
        input["shared_values"] = torch.stack(input["shared_values"])

        predctions = call_model(model, input)

        for i, e in enumerate(new_edges):
            edge_predictions[e] = (predctions[i].item(), input["left_values"][i][1].item() + input["right_values"][i][1].item() - edges[e] * 2)

        #for e in new_edges:
        #    input = {"left_values":tensors[e[0]], "right_values": tensors[e[1]], "shared_values": get_single_value_tensor(edges[e])}
        #    edge_predictions[e] = (call_model(model, input).item(), input["left_values"][1].item() + input["right_values"][1].item() - edges[e] * 2)

        prediction_times += time.time()

        step, prediction = min(edge_predictions.items(), key=lambda x:x[1][0])# / x[1][1])
        path.append(step)

        if print_sizes:
            print(prediction)

        if data is not None:
            data["path_data"]["size_predictions"].append(prediction)

        cleanup_time -= time.time()

        i = edges.pop(step)
        edge_predictions.pop(step)

        tensor = tensors.pop(step[0])
        tensor = tensors[step[1]] + tensor
        tensor[0] = prediction[0]
        tensor[1] -= i * 2
        tensors[step[1]] = tensor

        index_sets[step[0]].remove(step[1])
        index_sets[step[1]].remove(step[0])

        for i in index_sets[step[0]]:
            key = (min(step[0], i), max(step[0], i))

            if key[0] != key[1] and step[1] != i:
                index_sets[i].add(step[1])
                index_sets[i].remove(step[0])
                index_sets[step[1]].add(i)

                n = edges.pop(key)
                edge_predictions.pop(key)

                k = key[1] if key[0] == step[0] else key[0]
                new_key = (min(k, step[1]), max(k, step[1]))

                if new_key in edges:
                    edges[new_key] += n
                else:
                    edges[new_key] = n
        cleanup_time += time.time()

        new_edges = [(min(step[1], i), max(step[1], i)) for i in index_sets[step[1]]]

    print(f"Prediction Time: {prediction_times}, Cleanup Time: {cleanup_time}")

    return path

def take_step(step, tensors, index_sets, edges, edge_predictions, prediction):
    i = edges.pop(step)
    edge_predictions.pop(step)

    tensor = tensors.pop(step[0])
    tensor = tensors[step[1]] + tensor
    tensor[0] = prediction
    tensor[1] -= i * 2
    tensors[step[1]] = tensor

    index_sets[step[0]].remove(step[1])
    index_sets[step[1]].remove(step[0])

    for i in index_sets[step[0]]:
        key = (min(step[0], i), max(step[0], i))

        if key[0] != key[1] and step[1] != i:
            index_sets[i].add(step[1])
            index_sets[i].remove(step[0])
            index_sets[step[1]].add(i)

            n = edges.pop(key)
            edge_predictions.pop(key)

            k = key[1] if key[0] == step[0] else key[0]
            new_key = (min(k, step[1]), max(k, step[1]))

            if new_key in edges:
                edges[new_key] += n
            else:
                edges[new_key] = n

    return [(min(step[1], i), max(step[1], i)) for i in index_sets[step[1]]] 

def choose_step(weight_func, edge_predictions, tree_node, path_bound, sample_num):
    weight_sum = 0
    weights = []

    for step, (prediction, bound) in edge_predictions.items():
        if step in tree_node:
            visits = tree_node[step]["visits"]
            path_prediction = tree_node[step]["path_prediction"]
        else:
            visits = 0
            path_prediction = -1

        weight = weight_func(prediction, bound, visits, path_prediction, path_bound, sample_num)
        weight_sum += weight
        weights.append((step, weight))

    step_value = random() * weight_sum
    weight_sum = 0

    step = None
    prediction = None
    for s, w in weights:
        weight_sum += w
        if weight_sum > step_value:
            step = s
            prediction, _ = edge_predictions[s]
            break

    return step, prediction

def sample_path(model, tensor_network, tree, weight_func, path_bound, sample_num, data, agg):
    path = []

    t = time.time()
    tensors, index_sets = get_tensors(tensor_network)
    data["tensor_time"].append(time.time() - t)

    t = time.time()
    edges = get_edges(index_sets, tensor_network)
    data["edge_time"].append(time.time() - t)
    
    new_edges = list(edges.keys())
    edge_predictions = {}

    tree_node = tree[0]
    model_time = []
    choice_time = []
    step_time = []
    input_time = []
    prediction_time = []
    item_time = []
    stack_time = []

    predicted_sizes = []

    while len(edges) > 0:

        t = time.time()
        input = {"left_values":[], "right_values": [], "shared_values": get_tensor([edges[e] for e in new_edges], transposed = True)}

        for e in new_edges:
            input["left_values"].append(tensors[e[0]])
            input["right_values"].append(tensors[e[1]])
        input_time.append(time.time() - t)

        t = time.time()
        input["left_values"] = torch.stack(input["left_values"])
        input["right_values"] = torch.stack(input["right_values"])
        stack_time.append(time.time() - t)

        t = time.time()
        predctions = call_model(model, input).tolist()
        model_time.append(time.time() - t)

        t = time.time()
        left = input["left_values"].tolist()
        right = input["right_values"].tolist()
        item_time.append(time.time() - t)

        t = time.time()
        for i, e in enumerate(new_edges):
            edge_predictions[e] = (predctions[i][0], left[i][1] + right[i][1] - edges[e] * 2)
        prediction_time.append(time.time() - t)

        t = time.time()
        step, prediction = choose_step(weight_func, edge_predictions, tree_node, path_bound, sample_num)
        choice_time.append(time.time() - t)

        predicted_sizes.append(prediction)

        path.append(step)

        if step not in tree_node:
            tree_node[step] = {"visits": 0, "child":len(tree), "path_prediction":float("inf")}
            tree.append({})

        tree_node = tree[tree_node[step]["child"]]

        t = time.time()
        new_edges = take_step(step, tensors, index_sets, edges, edge_predictions, prediction)        
        step_time.append(time.time() - t)

    data["model_time"].append(model_time)
    data["choice_time"].append(choice_time)
    data["step_time"].append(step_time)
    data["input_time"].append(input_time)
    data["prediction_time"].append(prediction_time)
    data["item_time"].append(item_time)
    data["stack_time"].append(stack_time)
    data["all_size_predictions"].append(predicted_sizes)

    if agg == "sum":
        value = sum(predicted_sizes)
    elif agg == "max":
        value = max(predicted_sizes)
    elif type(agg) == float:
        value = max([v * (i + 1) ** agg for i, v in enumerate(predicted_sizes)])

    return path, value

def back_propegate(path, value, tree):
    node = tree[0]
    for p in path:
        edge = node[p]
        edge["visits"] += 1

        if value < edge["path_prediction"]:
            edge["path_prediction"] = value
        node = tree[edge["child"]]


def get_tree_search_path(model, tensor_network, weight_func, data, settings):
    max_time = settings["max_time"]
    path_bound = len(tensor_network.outer_inds()) * 2
    best_value = float("inf") #path_bound #changed to negative for testing!!!
    tree = [{}]
    start_time = time.time()

    sample_num = 0

    data["sample_time"] = []
    data["propagation_time"] = []
    data["model_time"] = []        # used in sample_path()
    data["choice_time"] = []       # used in sample_path()
    data["step_time"] = []         # used in sample_path()
    data["input_time"] = []        # used in sample_path()
    data["prediction_time"] = []   # used in sample_path()
    data["tensor_time"] = []       # used in sample_path()
    data["edge_time"] = []         # used in sample_path()
    data["item_time"] = []         # used in sample_path()
    data["stack_time"] = []         # used in sample_path()
    data["all_size_predictions"] = [] # used in sample_path()

    data["size_predictions"] = []
    paths = []

    while time.time() - start_time < max_time:
        t = time.time()
        latest_path, latest_value = sample_path(model, tensor_network, tree, weight_func, path_bound, sample_num, data, settings["aggregation"])
        data["sample_time"].append(time.time() - t)

        data["size_predictions"].append([latest_value])
        paths.append(latest_path)

        t = time.time()
        back_propegate(latest_path, latest_value, tree)
        data["propagation_time"].append(time.time() - t)
        
        if latest_value < best_value: #Changed from < to > for testing!!!!
            best_value = latest_value
            data["chosen_sample"] = sample_num

        print(f"Sample: {sample_num}, latest value: {latest_value}, best value: {best_value}")

        sample_num += 1

    #data["chosen_sample"] = int(len(paths) * random())
    return paths[data["chosen_sample"]]

