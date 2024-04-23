import plotting_util as pu
import file_util as fu
import math
from datetime import datetime as dt
from enum import Enum
from datetime import datetime
import numpy as np
from scipy import stats

class Variables(Enum):
    LEARNING_RATE = "Learning Rate lr"
    DEPTH = "Depth d"
    DROPOUT_PROBABILITY = "Dropout Probability p"
    HIDDEN_SIZE = "Hidden Layer Size s"
    BATCH_SIZE = "Batch Size log_2(|batch|)"
    WEIGHT_DECAY = "Weight Decay"
    LOSS = "Loss"
    EPOCHS = "Epoch num"
    LR_DECAY = "Learning Rate Decay lr_d"
    LR_DECAY_SPEED = "Learning Rate Decay Speed lr_s"
    TIME = "Training Time [s/epoch]"
    TIME_TOTAL = "Total Training Time"

def avg_last(l, num):
    return sum(l[-num:])/(num)

def avg(l):
    return sum(l)/len(l)

def get_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    n = len(data)
    stderr = stats.sem(data)
    interval = stderr * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - interval, mean + interval

def without_batch_data(data):
    out_data = {}
    for key in data:
        if key != "batch_data":
            out_data[key] = data[key]
    return out_data

if __name__ == "__main__":
    experiment_path = "experiment_data/tdd_mk_VI_depth"
    data = fu.load_all_json(experiment_path, without_batch_data)

    keys = {Variables.LEARNING_RATE: "lr", Variables.DEPTH:"depth", Variables.DROPOUT_PROBABILITY:"dropout_probability", 
            Variables.HIDDEN_SIZE:"hidden_size", Variables.BATCH_SIZE:"batch_size", Variables.WEIGHT_DECAY: "weight_decay",
            Variables.LR_DECAY: "lr_decay", Variables.LR_DECAY_SPEED: "lr_decay_speed"}
    x_axis = Variables.DEPTH
    y_axis = Variables.LEARNING_RATE
    z_axis = Variables.LOSS

    use_axies = 1

    log10_variables = [Variables.WEIGHT_DECAY]
    log2_variables = [Variables.HIDDEN_SIZE, Variables.BATCH_SIZE]

    x = [d[keys[x_axis]] for d in data]
    x = sorted(list(set(x)))

    y = [d[keys[y_axis]] for d in data]
    y = sorted(list(set(y)))

    def amm(func):
        ls = [[], [], []]
        for xv in x:
            l = [func(d) for d in data if d[keys[x_axis]] == xv]
            mean, minimum, maximum = get_confidence_interval(l)
            ls[0].append(mean)
            ls[1].append(minimum)
            ls[2].append(maximum)

        return ls

    if use_axies == 1:
        if z_axis == Variables.LOSS:
            d1_data = amm(lambda d : min(d["val_loss"]))
        elif z_axis == Variables.EPOCHS:
            d1_data = amm(lambda d : len(d["val_loss"]))
        elif z_axis == Variables.TIME:
            d1_data = amm(lambda d : (datetime.fromisoformat(d["end_time"]).timestamp() - datetime.fromisoformat(d["begin_time"]).timestamp()) / len(d["loss"]))
        elif z_axis == Variables.TIME_TOTAL:
            d1_data = amm(lambda d : datetime.fromisoformat(d["end_time"]).timestamp() - datetime.fromisoformat(d["begin_time"]).timestamp())
    elif use_axies == 2:
        if z_axis == Variables.LOSS:
            grouped_data = [[[min(min([min(d["val_loss"]) for d in data if d[keys[x_axis]] == xv and d[keys[y_axis]] == yv]), 1200) for yv in y] for xv in x]]
        elif z_axis == Variables.EPOCHS:
            grouped_data = [[[min(avg([len(d["val_loss"]) for d in data if d[keys[x_axis]] == xv and d[keys[y_axis]] == yv]), 1200) for yv in y] for xv in x]]

    if x_axis in log10_variables:
        x = [math.log10(v) for v in x]
    elif  x_axis in log2_variables:
        x = [math.log2(v) for v in x]

    if y_axis in log10_variables:
        y = [math.log10(v) for v in y]
    elif  y_axis in log2_variables:
        y = [math.log2(v) for v in y]

    labels = ["Epoch num" if z_axis == Variables.EPOCHS else "Val loss"]

    if use_axies == 1:
        pu.plot_line_series_2d([x, x, x], d1_data, ["mean", "mean-ci", "mean+ci"], x_label=x_axis.value, y_label=z_axis.value, legend= True)
    elif use_axies == 2:
        pu.plotSurface(grouped_data, z_axis.value, x, x_axis.value, y, y_axis.value, 1, labels)

    #print(grouped_data)
#
    #
    #avg = 10
    #pu.plot_line_series_2d([x,x], 
    #                       [[sum([avg_last(d["loss"], avg) for d in gd]) / len(gd)for gd in grouped_data], 
    #                       [sum([avg_last(d["val_loss"], avg) for d in gd]) / len(gd) for gd in grouped_data]], ["loss", "val loss"], legend= True)

    optimal = min(data, key = lambda d: min(d["val_loss"]))
    optimal = fu.load_single_json(fu.get_path(experiment_path + "/" + optimal["run_name"] + ".json"))

    error = []
    outputs = []
    targets = []

    for i, output in enumerate(optimal["batch_data"][0]):
        output = output[0]
        #left = optimal[1][i]
        #right = optimal[2][i]
        #shared = optimal[3][i]
        target = optimal["batch_data"][4][i][0]
        error.append([(output - target)])
        outputs.append(output)
        targets.append(target)

    def exponmentiate(l):
        return [[2**v[0]] for v in l]

    #pu.plotPoints2d((optimal["batch_data"][4]), error, "Target","Error", legend=False, marker_size=2)
    pu.plotPoints2d([outputs], [targets], "Output", "Target", legend=False, marker_size=1)

    x = [i for i, _ in enumerate(optimal["loss"])]
    #avg = 10

    pu.plot_line_series_2d([x,x], [[min(3000, i) for i in optimal["loss"]], [min(3000, i) for i in  optimal["val_loss"]]], ["loss", "val loss"], x_label="Epochs", y_label="Loss", legend= True)
    print(f"Optimal: {optimal['run_name']}, Lr: {(optimal['lr'])}, Lr Decay: {optimal['lr_decay']}, Lr Decay Speed: {optimal['lr_decay_speed']}, Depths: {optimal['depth']}, Dropout: {math.log2(optimal['dropout_probability'])},"+
          f" Hidden Size: {optimal['hidden_size']}, Batch Size: {optimal['batch_size']}, Val Loss: {min(optimal['val_loss'])}")

    #print(f"Loss: {sum([d['loss'][-1] for d in data])/len(data)}, Val Loss: {sum([d['val_loss'][-1] for d in data])/len(data)}")


    
    #pu.plot_line_series_2d([x,x], [[avg_last(d["loss"], avg) for d in data], [avg_last(d["val_loss"], avg) for d in data]], ["loss", "val loss"], legend= True)
    
    max_cut = 100
    epoch_first = 10

    #pu.plotSurface([[[min(l,max_cut) for l in d["loss"][epoch_first:]] for d in data], [[min(l,max_cut) for l in d["val_loss"][epoch_first:]] for d in data]], 
    #               "Loss", x, "Lr log10(lr)", [i for i in range(epoch_first, len(data[0]["loss"]))], "Epoch e", 2, ["Loss", "Val Loss"])