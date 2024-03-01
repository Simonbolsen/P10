import plotting_util as pu
import file_util as fu
import math
from datetime import datetime as dt
from enum import Enum

class Variables(Enum):
    LEARNING_RATE = "Learning Rate lr"
    DEPTH = "Depth d"
    DROPOUT_PROBABILITY = "Dropout Probability p"
    HIDDEN_SIZE = "Hidden Layer Size s"
    BATCH_SIZE = "Batch Size"
    WEIGHT_DECAY = "Weight Decay"


def avg_last(l, num):
    return sum(l[-num:])/(num)

def avg(l):
    return sum(l)/len(l)

if __name__ == "__main__":
    data = fu.load_all_json("experiment_data/tdd_training_3")

    keys = {Variables.LEARNING_RATE: "lr", Variables.DEPTH:"depth", Variables.DROPOUT_PROBABILITY:"dropout_probability", 
            Variables.HIDDEN_SIZE:"hidden_size", Variables.BATCH_SIZE:"batch_size", Variables.WEIGHT_DECAY: "weight_decay"}
    x_axis = Variables.LEARNING_RATE
    y_axis = Variables.DEPTH

    log10_variables = [Variables.LEARNING_RATE, Variables.WEIGHT_DECAY]
    log2_variables = [Variables.HIDDEN_SIZE, Variables.BATCH_SIZE]

    x = [d[keys[x_axis]] for d in data]
    x = sorted(list(set(x)))

    y = [d[keys[y_axis]] for d in data]
    y = sorted(list(set(y)))

    grouped_data = [[[avg([min(d["val_loss"]) for d in data if d[keys[x_axis]] == xv and d[keys[y_axis]] == yv]) for yv in y] for xv in x]]

    if x_axis in log10_variables:
        x = [math.log10(v) for v in x]
    elif  x_axis in log2_variables:
        x = [math.log2(v) for v in x]

    if y_axis in log10_variables:
        y = [math.log10(v) for v in y]
    elif  y_axis in log2_variables:
        y = [math.log2(v) for v in y]

    pu.plotSurface(grouped_data, "Loss", x, x_axis.value, y, y_axis.value, 1, ["Val Loss"])

    #print(grouped_data)
#
    #
    #avg = 10
    #pu.plot_line_series_2d([x,x], 
    #                       [[sum([avg_last(d["loss"], avg) for d in gd]) / len(gd)for gd in grouped_data], 
    #                       [sum([avg_last(d["val_loss"], avg) for d in gd]) / len(gd) for gd in grouped_data]], ["loss", "val loss"], legend= True)

    optimal = min(data, key = lambda d: min(d["val_loss"]))

    error = []

    for i, output in enumerate(optimal["batch_data"][0]):
        output = output[0]
        #left = optimal[1][i]
        #right = optimal[2][i]
        #shared = optimal[3][i]
        target = optimal["batch_data"][4][i][0]
        error.append([(output - target)])

    def exponmentiate(l):
        return [[2**v[0]] for v in l]

    #pu.plotPoints2d((optimal["batch_data"][4]), error, "Target","Error", legend=False, marker_size=2)
    #pu.plotPoints2d((optimal["batch_data"][0]), (optimal["batch_data"][4]), "Output", "Target", legend=False, marker_size=2)

    x = [i for i, _ in enumerate(optimal["loss"])]
    #avg = 10

    pu.plot_line_series_2d([x,x], [[min(700, i) for i in optimal["loss"]], [min(700, i) for i in  optimal["val_loss"]]], ["loss", "val loss"], x_label="Epochs", y_label="Loss", legend= True)
    print(f"Lr: {math.log10(optimal['lr'])}, Depths: {optimal['depth']}, Dropout: {optimal['dropout_probability']},"+
          f" Hidden Size: {optimal['hidden_size']}, Batch Size: {optimal['batch_size']}, Val Loss: {min(optimal['val_loss'])}")

    #print(f"Loss: {sum([d['loss'][-1] for d in data])/len(data)}, Val Loss: {sum([d['val_loss'][-1] for d in data])/len(data)}")


    
    #pu.plot_line_series_2d([x,x], [[avg_last(d["loss"], avg) for d in data], [avg_last(d["val_loss"], avg) for d in data]], ["loss", "val loss"], legend= True)
    
    max_cut = 100
    epoch_first = 10

    #pu.plotSurface([[[min(l,max_cut) for l in d["loss"][epoch_first:]] for d in data], [[min(l,max_cut) for l in d["val_loss"][epoch_first:]] for d in data]], 
    #               "Loss", x, "Lr log10(lr)", [i for i in range(epoch_first, len(data[0]["loss"]))], "Epoch e", 2, ["Loss", "Val Loss"])