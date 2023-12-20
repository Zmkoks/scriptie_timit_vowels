import numpy as np
import pandas as pd
from information import *
from mfcc_to_df import run_main
from input_classifiers import run_delta, select_frames
from select_phones import select_vowels, expand_sel_from_array
from networking import run_networking
from output import run_output
from data import reform_data

import os

if not reduce_groups:
    group_data={}


def get_end_col(dic_info):
    end = dic_info['n_mels']
    if dic_info['double_delta']:
        end = end*3
    elif dic_info['delta']:
        end = end*2
    end = end + 7
    return(end)

def check_for_raw_data(dic_info):
    ######## part one: get the raw data
    ### makes mfcc from .wav files, adds metadata from phn files, get delta and double delta:
    name_data = "_".join([str(dic_info["n_mels"]), str(dic_info["hoplength"]), str(dic_info["framelength"])])
    mfcc_path = os.path.join(data_path, name_data)
    attempt_path = os.path.join(mfcc_path, "attempt_" + dic_info['poging'])
    if not os.path.exists(attempt_path):
        os.mkdir(attempt_path)

    os.chdir(data_path)
    ############## check if data original from mfcc already exist (same mels, hoplength, framelength):
    if not os.path.exists(mfcc_path):
        run_main(name_data, tmit_path, hoplength=dic_info["hoplength"], framelength=dic_info["framelength"],
                 n_mels=dic_info["n_mels"], groups = group_data)
        os.mkdir(mfcc_path)
        run_delta(mfcc_path, mfcc_path)
        select_frames(mfcc_path, mfcc_path + "/middles_index", before=0, after=0, save_index=True)
        print('data_gathering_complete')

    os.chdir(mfcc_path)
    ########### make file with indices that point to 'select_frames':
    sel_frames = dic_info['select_frames']
    if sel_frames:
        expand_sel_from_array(sel_frames, attempt_path)

    ########### check if we already have selected on phones and have the data:
    for i in dic_info['selection']:
        if not os.path.isfile(i + "_TRAIN.txt"):
            select_vowels(mfcc_path, i, all_info[i], save_index=True)
    return mfcc_path

def prepare_data(dic_info, mfcc_path, set):
    attempt_path = os.path.join(mfcc_path, "attempt_" + dic_info['poging'])

    big_data = pd.read_csv(mfcc_path + '_' + set + '.txt', header=None)
    big_data = big_data.iloc[:, 0:get_end_col(dic_info)]

    selected_array = []
    if dic_info['select_frames']:
        selected_array = pd.read_csv(attempt_path + '/sel_' + set + '.txt', header=None)
    else:
        selected_array = list(big_data.index)

    if 'wide' in dic_info['type']:
        middles = pd.read_csv(mfcc_path + '/middles_index_' + set + '.txt', header=None)
        middles = list(middles[0].values)
        big_data = reform_data(big_data, dic_info, middles)
    middles = list(big_data.index)
    selected_array = np.intersect1d(middles, selected_array)

    os.chdir(attempt_path)
    size = dic_info['n_mels']
    if dic_info['double_delta']:
        size = size * 3
    elif dic_info['delta']:
        size = size * 2
    return(big_data, selected_array,attempt_path, size)




def do_network_stuff(dic_info, mfcc_path):
    ### run network
    ### save an information.txt with all info from dic_info
    ### saves models with selected each phone/group of phones in all_info
    big_data, selected_array,attempt_path, size = prepare_data(dic_info, mfcc_path, 'TRAIN')

    for i in dic_info['selection']:
        if not os.path.exists(attempt_path + "/" + i + "_model_" + dic_info["poging"]):
            data_from = pd.read_csv(mfcc_path + '/' + i + '_TRAIN.txt', header=None)

            data_array = list(np.intersect1d(data_from[0].values, selected_array))
            print(str(data_array[:10]) + '\t'+ str(data_array[-10:]))
            data = big_data.loc[data_array,:]

            run_networking(data, i + "_model_" + dic_info["poging"],
                           del_dia=dic_info["delete_dialects"],
                           del_sex=dic_info["delete_gender"],
                           classifier_col=dic_info["network_classes"],
                           epoch = dic_info['epoch'],
                           batchsize = dic_info['batch_size'],
                           ntype= dic_info['type'],
                           n_cols=size,
                           sel = dic_info['select_frames'])
        print(i + " network done")
    with open("information.txt", 'w') as f:
        for i in dic_info:
            f.write(i + "\t" + str(dic_info[i]))
            f.write('\n')
    return attempt_path


def do_output_stuff(dic_info, mfcc_path, attempt_path):
    ### run the network with the test-set of the same selection-data,
    ### save a csv with output: is a table with all precision/recall/accuracy/f1-score
    ### save a confusion matrix
    big_data, selected_array, attempt_path, size = prepare_data(dic_info, mfcc_path, 'TEST')

    if os.path.exists("output_m.txt") or not os.path.exists("output_m.txt"):
        output = {}
        output_m = {}
        for i in dic_info['selection']:
            name_path = mfcc_path + "/" + i + "_TEST.txt"
            data_from = pd.read_csv(name_path, header=None)

            data_array = list(np.intersect1d(data_from[0].values, selected_array))
            data = big_data.loc[data_array,]

            model_name = i + "_model_" + dic_info["poging"]
            c, m = run_output(model_name, data,
                           del_dia=dic_info["delete_dialects"],
                           del_sex=dic_info["delete_gender"],
                           classes =dic_info['network_classes'],
                           size = size,
                           sel = dic_info['select_frames'])
            #c["mcof"] = m
            output[i] = pd.DataFrame(c)

        output_df = pd.concat(output, axis=1)
        output_df.to_csv("output.txt")
    if not os.path.exists(output_path + "/" + "model_" + dic_info["poging"]):
        os.symlink(attempt_path, output_path + "/" + "model_" + dic_info["poging"])


def get_all(dic_info, save_as_index=False):
    print("start")
    big_folder = check_for_raw_data(dic_info)
    # name_data = "_".join([str(dic_info["n_mels"]), str(dic_info["hoplength"]), str(dic_info["framelength"])])
    # big_folder = os.path.join(data_path, name_data)
    # small_folder = os.path.join(big_folder, "attempt_" + dic_info['poging'])
    # os.chdir(small_folder)
    print("big_folder_done")
    small_folder = do_network_stuff(dic_info, big_folder)

    print("small_folder_done")
    do_output_stuff(dic_info, big_folder, small_folder)


def check_before(dic_info, save_as_index=False):
    os.chdir(output_path)
    model_name = 'model_' + dic_info['poging']
    if not os.path.exists(model_name):
        get_all(dic_info, save_as_index)
    else:
        os.chdir(output_path + '/' + model_name)
        info_df = pd.read_csv('information.txt', header=None, index_col=0, sep='\t')
        info_dic_old = info_df.to_dict()[1]
        diff = []
        for inf in info_dic_old:
            dic = str(dic_info[inf])
            if info_dic_old[inf] != dic:
                diff.append(inf)
        if len(diff) > 0:
            print('WARNING: a model by that name already exist, continue? \n'
                  'these are the differences: {}'.format(diff))
            inp = ''
            while inp != 'N' and inp != 'Y':
                inp = input('For more information on one, '
                            'give string. To quit, "N", and to continue "Y"')
                if inp in diff:
                    print('old value: {}, new value: {}'.format(info_dic_old[inp], dic_info[inp]))
            if inp == 'N':
                return 0
            else:
                get_all(dic_info, save_as_index)
        else:
            print('WARNING, YOU ALREADY HAVE RUN THIS, WITH THE SAME THINGS, RUNNING OUTPUT AGAIN')
            get_all(dic_info, save_as_index)


check_before(dic_info, save_as_index=True)

def make_fancy_output():
    wdir = output_path
    os.chdir(output_path)

    from output_png import imp_confusion_txt
    from to_excel import main
    imp_confusion_txt()
    main()

make_fancy_output()
