# -*- coding: utf-8 -*-
# %% LIBRARIES AND HYPER-PARAMETERS
from hyper_parameters import setup_hparams

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import warnings

warnings.filterwarnings("ignore")

hps = setup_hparams(sys.argv[1:]) # hyper-parameters
data_split, n_splits, test = hps["data_split"], hps["n_splits"], hps["test"] # model variables
classes_names = ["girolando", "holandes"] # classes names
results_folder = "results" # path to the results folder

# path to the plots folder
plots_folder = os.path.join(os.getcwd(),
                            "plots",
                            "test_{}".format(test))

if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# %% TRAINING HISTORY

tab_hist = pd.read_csv("{}/histories_{}_{}.csv".format(results_folder, test, data_split))
df_hist = pd.DataFrame(tab_hist)

tab_val_acc_class = pd.read_csv("{}/val_acc_class_history_{}_{}.csv".format(results_folder, test, data_split))
df_val_acc_class_hist = pd.DataFrame(tab_val_acc_class)

# %% PLOT FUNCTION

def training_results(split):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"]
    })
    
    mpl.style.use("seaborn-v0_8-whitegrid")
    
    epochs_range = range(1, len(df_hist["train_acc_hist_split_{}".format(split)])+1)
    
    plt.figure(figsize = (30, 30))
    
    plt.plot(epochs_range, df_hist["train_acc_hist_split_{}".format(split)], label = "Training Accuracy")
    plt.plot(epochs_range, df_hist["val_acc_hist_split_{}".format(split)], label = "Validation Accuracy")
    plt.xlim(1, len(epochs_range))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xlabel("Epochs", labelpad = 72, fontsize = 72)
    plt.xticks(fontsize = 60)
    plt.ylabel("Accuracy", labelpad = 72, fontsize = 72)
    plt.yticks(fontsize = 60)
    plt.legend(loc = "best", fontsize = 72, frameon = True)
    
    plt.savefig("{}/acc_{}_{}_{}.pdf".format(plots_folder, test, data_split, split), bbox_inches = "tight", pad_inches = 0)
    
    plt.figure(figsize = (30, 30))
    
    plt.plot(epochs_range, df_hist["train_loss_hist_split_{}".format(split)], label = "Training Loss")
    plt.plot(epochs_range, df_hist["val_loss_hist_split_{}".format(split)], label = "Validation Loss")
    plt.xlim(1, len(epochs_range))
    plt.ylim(0)
    plt.grid(True, color = "white")
    plt.xlabel("Epochs", labelpad = 72, fontsize = 72)
    plt.xticks(fontsize = 60)
    plt.ylabel("Loss", labelpad = 72, fontsize = 72)
    plt.yticks(fontsize = 60)
    plt.legend(loc = "best", fontsize = 72, frameon = True)
    
    plt.savefig("{}/loss_{}_{}_{}.pdf".format(plots_folder, test, data_split, split), bbox_inches = "tight", pad_inches = 0)
    
    plt.figure(figsize = (30, 30))
    
    for c in range(len(classes_names)):
        plt.plot(epochs_range, df_val_acc_class_hist["val_acc_hist_{}_split_{}".format(classes_names[c], split)], label = "Val Acc - {}".format(classes_names[c]))
    
    plt.xlim(1, len(epochs_range))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xlabel("Epochs", labelpad = 72, fontsize = 72)
    plt.xticks(fontsize = 60)
    plt.ylabel("Accuracy", labelpad = 72, fontsize = 72)
    plt.yticks(fontsize = 60)
    leg = plt.legend(loc = "best", fontsize = 54, frameon = True)
    
    for line in leg.get_lines():
        line.set_linewidth(6)

    plt.savefig("{}/acc_val_class_{}_{}_{}.pdf".format(plots_folder, test, data_split, split), bbox_inches = "tight", pad_inches = 0)

for split in range(1, n_splits+1):
    training_results(split)
