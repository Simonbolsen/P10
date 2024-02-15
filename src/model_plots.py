import plotting_util as pu
import file_util as fu
import math

def avg_last(l, num):
    return sum(l[-num:])/(num)

if __name__ == "__main__":
    data = fu.load_all_json("experiment_data/retraining")

    x = [d["lr"] for d in data]
    #x = sorted(list(set(x)))
    #grouped_data = [[d for d in data if d["lr"] == lr] for lr in x]
#
    #x = [math.log10(v) for v in x]
    #avg = 10
    #pu.plot_line_series_2d([x,x], 
    #                       [[sum([avg_last(d["loss"], avg) for d in gd]) / len(gd)for gd in grouped_data], 
    #                       [sum([avg_last(d["val_loss"], avg) for d in gd]) / len(gd) for gd in grouped_data]], ["loss", "val loss"], legend= True)

    data = sorted(data, key = lambda d: d["lr"])
    x = [math.log10(d["lr"]) for d in data]
    avg = 10

    pu.plot_line_series_2d([x,x], [[d["loss"][-1] for d in data], [d["val_loss"][-1] for d in data]], ["loss", "val loss"], x_label="Learning Rate Lr", y_label="Loss", legend= True)

    print(f"Loss: {sum([d['loss'][-1] for d in data])/len(data)}, Val Loss: {sum([d['val_loss'][-1] for d in data])/len(data)}")

    
    #pu.plot_line_series_2d([x,x], [[avg_last(d["loss"], avg) for d in data], [avg_last(d["val_loss"], avg) for d in data]], ["loss", "val loss"], legend= True)
    
    max_cut = 100
    epoch_first = 10

    #pu.plotSurface([[[min(l,max_cut) for l in d["loss"][epoch_first:]] for d in data], [[min(l,max_cut) for l in d["val_loss"][epoch_first:]] for d in data]], 
    #               "Loss", x, "Lr log10(lr)", [i for i in range(epoch_first, len(data[0]["loss"]))], "Epoch e", 2, ["Loss", "Val Loss"])