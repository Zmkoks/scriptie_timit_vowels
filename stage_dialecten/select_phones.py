import os
import numpy as np
import pandas as pd
from information import vowels




def import_data(path, isin=None, save_index = False):
    if isin is None:
        isin = vowels
    data = pd.read_csv(path, header=None)
    data = data.loc[data[5].isin(isin), :]
    if save_index:
        data = list(data.index)
    return data


def select_vowels(name="all_data", new_name="selected", isin=None, save_index = False):
    print('selecting_vowels ' + name)
    test_name = name + "_TEST.txt"
    test_new_name = "_".join([new_name, "TEST.txt"])
    selected = import_data(test_name, isin, save_index)
    np.savetxt(test_new_name, selected, delimiter=",", fmt="%s")
    train_name = name + "_TRAIN.txt"
    train_new_name = "_".join([new_name, "TRAIN.txt"])
    selected = import_data(train_name, isin, save_index)
    np.savetxt(train_new_name, selected, delimiter=",", fmt="%s")

def expand_sel_from_array(sel_frames, output_dir, save = True):
    before = 1
    after = 1
    if type(sel_frames) is list:
        before = sel_frames[0]
        after = sel_frames[1]
    elif type(sel_frames) is int:
        before = sel_frames
        after = sel_frames
    for i in ["_TEST.txt", "_TRAIN.txt"]:
        new_array = []
        array_middles = list(pd.read_csv('middles_index' + i, header=None)[0].values)
        for middle in array_middles:
            for b in range(before, 0, -1):
                new_array.append(middle - b)
            new_array.append(middle)
            for a in range(1, after + 1, 1):
                new_array.append(middle + a)
        if save:
            np.savetxt(output_dir+'/sel'+i, new_array, delimiter=',', fmt="%s")
        else:
            return(new_array)
