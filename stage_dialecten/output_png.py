import keras
import pandas as pd
import numpy as np
import sklearn.metrics
from colorspacious import cspace_converter
import os
attempts = []

wdir = "/home/zjoske/Documents/stage_outcome"
os.chdir("/home/zjoske/Documents/stage_outcome")

for i in os.listdir():
    if os.path.isdir(i):
        attempts.append(i)

def make_png(pathname, file_name):
    print(file_name)
    df = pd.read_csv(pathname + "/"+ file_name, index_col = 0)

    import matplotlib.pyplot as plt
    import seaborn as sn
    fig, ax = plt.subplots()
    sn.heatmap(df, ax = ax)
    ax.xaxis.set_label_text('predicted')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_label_text('true')
    plt.title(file_name[:-15], y = -0.1)
    name = file_name[:-4]+ ".png"
    plt.savefig(name)
    plt.show()


def imp_confusion_txt():
    for a in attempts:
        pathname = os.path.join(wdir, a)
        selection = []
        for file in os.listdir(pathname):
            print(file)
            if "con_matrix.txt" in file:
                make_png(pathname,file)




