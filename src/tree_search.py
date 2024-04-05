import math
import time
import torch
from tqdm import tqdm
from random import random
from tdd_nn import get_single_value_tensor
from tdd_nn import get_tensors
from tdd_nn import call_model

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

        for e in new_edges:
            input = {"left_values":tensors[e[0]], "right_values": tensors[e[1]], "shared_values": get_single_value_tensor(edges[e])}
            edge_predictions[e] = (call_model(model, input).item(), input["left_values"][1].item() + input["right_values"][1].item() - edges[e] * 2)

        prediction_times += time.time()

        step, prediction = min(edge_predictions.items(), key=lambda x:x[1][0] / x[1][1])
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

    return step, prediction

def sample_path(model, tensor_network, tree, weight_func, path_bound, sample_num):
    path = []
    path_prediction = 0

    tensors, index_sets = get_tensors(tensor_network)
    edges = get_edges(index_sets, tensor_network)
    
    new_edges = list(edges.keys())
    edge_predictions = {}

    tree_node = tree[0]

    while len(edges) > 0:

        for e in new_edges:
            input = {"left_values":tensors[e[0]], "right_values": tensors[e[1]], "shared_values": get_single_value_tensor(edges[e])}
            edge_predictions[e] = (call_model(model, input).item(), input["left_values"][1].item() + input["right_values"][1].item() - edges[e] * 2)

        step, prediction = choose_step(weight_func, edge_predictions, tree_node, path_bound, sample_num)

        if prediction > path_prediction:
            path_prediction = prediction

        path.append(step)

        if step not in tree_node:
            tree_node[step] = {"visits": 0, "child":len(tree), "path_prediction":float("inf")}
            tree.append({})

        tree_node = tree[tree_node[step]["child"]]

        new_edges = take_step(step, tensors, index_sets, edges, edge_predictions, prediction)        

    return path, path_prediction

def back_propegate(path, value, tree):
    node = tree[0]
    for p in path:
        edge = node[p]
        edge["visits"] += 1

        if value < edge["path_prediction"]:
            edge["path_prediction"] = value


def get_tree_search_path(model, tensor_network, weight_func, time_limit = 60):

    best_path = []
    path_bound = len(tensor_network.out_inds) * 2
    best_value = path_bound
    tree = [{}]
    start_time = time.time()

    progress_bar = tqdm()
    sample_num = 1

    while time.time() - start_time < time_limit:
        latest_path, latest_value = sample_path(model, tensor_network, tree, weight_func, path_bound, sample_num)
        back_propegate(latest_path, latest_value, tree)

        if latest_value < best_value:
            best_path = latest_path
            best_value = latest_value

        sample_num += 1
        progress_bar.update(1)
    return best_path

