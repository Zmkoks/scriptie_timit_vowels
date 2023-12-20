from information import dic_info
from select_phones import expand_sel_from_array
import pandas as pd
import itertools
import numpy as np
information = dic_info

#from this format:
# meta-info - 'a' - mfcc 1, mfcc 2, mfcc 3 ...
# meta-info - 'a' - mfcc 1, mfcc 2, mfcc 3 ...
# meta-info - 'a' - mfcc 1, mfcc 2, mfcc 3 ...
# meta-info - 'b' - mfcc 1, mfcc 2, mfcc 3 ...
# meta-info - 'b' - mfcc 1, mfcc 2, mfcc 3 ...
# meta-info - 'b' - mfcc 1, mfcc 2, mfcc 3 ...
#
# to this format:
# meta-info - 'a' - mfcc 1, mfcc 2, mfcc 3 ... mfcc 1, mfcc 2, mfcc 3 ... mfcc 1, mfcc 2, mfcc 3 ...
# meta-info - 'b' - mfcc 1, mfcc 2, mfcc 3 ... mfcc 1, mfcc 2, mfcc 3 ... mfcc 1, mfcc 2, mfcc 3 ...
def repeated(to_be, times):
    arr = []
    if type(to_be) is int:
        arr = list(itertools.repeat(to_be, times))
    if type(to_be) is list:
        arr = list(itertools.chain.from_iterable(itertools.repeat(x,times) for x in to_be))
    return arr

def reform_data(df, dic_info, middles):
    sel = dic_info['select_frames']
    if sel is None or sel == False:
        print('select_frames must be int or list for cnn_long!')
        return(None)
    if type(sel) == int:
        sel = [sel, sel]
    df_new = df.iloc[middles, :6]
    range_all = list(range(-sel[0], sel[1]+1))
    for i in range_all:
        try:
            df_new_slice = df.iloc[[var + i for var in middles], 7:].set_axis(middles, axis = 'index')
        except IndexError:
            df_new_slice = df.iloc[[var + i for var in middles[1:-1]], 7:].set_axis(middles[1:-1], axis = 'index')
        df_new = pd.concat([df_new, df_new_slice], axis = 1)

    return df_new


















