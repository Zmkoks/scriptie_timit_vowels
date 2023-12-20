import pandas as pd
import warnings
import numpy as np

def run_delta(name_basic, name_delta, delta = True, double_delta = True):
    if double_delta and not delta:
        warnings.warn("you cannot have double delta without delta, delta will be added as well")
    for i in ["_TEST.txt","_TRAIN.txt"]:
        df = pd.read_csv(name_basic + i, header = None)
        if delta and not double_delta:
            df = get_delta(df)
        if double_delta:
            df = get_delta_and_double_delta(df)
        np.savetxt(name_delta+i,df, delimiter=",", fmt="%s")


def get_delta(all_data_table):
    # select only columns with data:
    data_frame = all_data_table.iloc[:, 7:]
    # delta, get differences between row x and row x+2, remove first row to
    # get the rows to match the correct original delta;
    # fill last row with nAn to make it even:
    df = data_frame.diff(periods=2)
    df = df.drop([0])
    df = df.append(pd.Series(), ignore_index=True)
    #join everything:
    all_data_table_delta = pd.concat([all_data_table, df], axis = 1)
    return (all_data_table_delta)



def get_delta_and_double_delta(all_data_table):
    #select only columns with data:
    data_frame = all_data_table.iloc[:,7:]
    #delta, get differences between row x and row x+2, remove first row to
    # get the rows to match the correct original delta;
    # fill last row with nAn to make it even:
    df = data_frame.diff(periods=2)
    df = df.drop([0])
    df = df.append(pd.Series(), ignore_index=True)

    # get double delta, delta of delta
    double_df = df.diff(periods=2)
    double_df = double_df.drop([0])
    double_df = double_df.append(pd.Series(), ignore_index=True)

    #join everything:
    all_data_table_delta = pd.concat([all_data_table, df, double_df], axis = 1)
    return (all_data_table_delta)



def get_info_index_phone_list(data):
    ## returns a list where each item is a list with:
    ## phone, frame/row start, frame/row end
    list_phone_index_info = []
    prev_phone = data.iloc[0, 5]
    start_phone = 0
    for index, row in data.iterrows():
        phone = row[5]
        if phone != prev_phone:
            list_phone_index_info.append([prev_phone, start_phone, index])
            prev_phone = phone
            start_phone = index

    list_phone_index_info.append([prev_phone, start_phone, len(data)])
    return list_phone_index_info


def select_middle_frames(path, before = 1, after = 1, save_index = False):
    ## returns dataframe with only the middle frame(s) of a certain sound/phone
    ## middle frame only -> middle_int = 0; middle int = how many frames before and after included
    data = pd.read_csv(path, header=None)
    info_list = get_info_index_phone_list(data)
    # middles:
    middles = []
    for i in info_list:
        middle_index = (i[2] + i[1]) // 2
        for j in range(before, 0, -1):
            middles.append(middle_index-j)
        middles.append(middle_index)
        for j in range(1, (after+1), 1):
            middles.append(middle_index+j)
    ## catch problem if the first/last frames fall out bounds
    try:
        middle_data = data.iloc[middles, :]
    except(IndexError):
        middles = middles[3:-3]
        middle_data = data.iloc[middles, :]
    if save_index:
        return middles
    all_data = middle_data
    return all_data


def select_frames(name = "all_data", new_name = "curated", before = 1, after = 1, save_index = False):
    print("selecting frames...")
    print(save_index)
    test_name =  name + "_TEST.txt"
    test_new_name = "_".join([new_name, "TEST.txt"])
    print(test_new_name)
    selected = pd.DataFrame(select_middle_frames(test_name, before, after, save_index))
    np.savetxt(test_new_name, selected, delimiter=",", fmt="%s")
    print("first_done")
    train_name =  name + "_TRAIN.txt"
    train_new_name = "_".join([new_name, "TRAIN.txt"])
    print(train_new_name)
    selected = pd.DataFrame(select_middle_frames(train_name, before, after, save_index))
    np.savetxt(train_new_name, selected, delimiter=",", fmt="%s")
    print("second_done")
