import os
import pandas as pd
import xlsxwriter
import numpy as np
import string
import math
wdir = "/home/zjoske/Documents/stage_outcome"
os.chdir("/home/zjoske/Documents/stage_outcome")
vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix',
          'axr', 'ax-h']

all_info = {'all_vowels': vowels,
            'all_i': ['ih', 'ix', 'iy'],
            'all_u': ['uh', 'uw', 'ux'],
            'all_a': ['aa', 'ae', 'ah'], 'all_mid': ['ax-h', 'ax', 'axr', 'eh', 'ay', 'er', 'ow'], 'just_i': ['iy'],
            'just_u': ['uw'], 'just_a': ['aa'], 'just_mid': ['ax']
            }


all_info_list = list(all_info)
with open("output_avg.txt", "w") as f:
    f.write("model")
    for i in all_info_list:
        f.write("," + i)
    f.write(",best,n class")

with open("sigma.txt", "w") as f:
    f.write("model")
    for i in all_info_list:
        f.write("," + i)
    f.write(",best,n class")

with open("st_sigma.txt", "w") as f:
    f.write("model")
    for i in all_info_list:
        f.write("," + i)
    f.write(",best,n class")

writer = pd.ExcelWriter('/media/zjoske/Seagate Expansion Drive/scriptie/output2.xlsx', engine='xlsxwriter')
attempts = []
for i in os.listdir():
    if os.path.isdir(i):
        attempts.append(i)

def clean_df(name_attempt):
    path = os.path.join(wdir, str(name_attempt))
    df = pd.read_csv(path+"/output.txt", header = None)
    df.iloc[0,0] = "selection"
    df.iloc[1,0] = "dialect"
    df = df.set_index(0)
    df = df.transpose()
    return df,path

def turn_df(df, key):
    ### selects only those columns with phone = key, further: columns = precision, recall, f1-score, support
    ### rows: selection, accuracy, macro avg, weighted avg + individual classifier classes
    df_new = df.loc[df.iloc[:,0]==key, :].iloc[:,1:]
    df_new = df_new.transpose()
    df_new = df_new.rename(columns = df_new.iloc[0]).drop(df_new.index[0])
    df_new = df_new.astype('float')
    df_new.loc[:,"selection"] = key
    df_new = df_new.transpose()
    df_index = list(df_new.index)
    df_index = [df_index[-1]] + df_index[-4:-1] + df_index[0:-4]
    df_new = df_new.reindex(df_index)
    return df_new

def write_dict(attempt, dic, chance, x = 0, name = 'sed.txt', mult = 1):
    new_row = []
    sizes = {}
    if not mult:
        mult = 1
    for i in all_info_list:
        if not dic[i][x]:
            new_row.append(None)
        else:
            new_row.append(float(dic[i][x])*mult)
            sizes[i] = dic[i][x]
    with open(name, "a+") as f:
        f.write("\n")
        f.write(attempt)
        for i in new_row:
            if i:
                f.write("," + str(round(i,2)))
            else:
                f.write(",")
        f.write("," + max(sizes, key = sizes.get))
        f.write("," + str(chance))


def make_empty_slot(df_1, info):
    for i in list(df_1.index):
        df_1.loc[i] = None
    df_1.loc["selection"] = info
    return df_1


def df_to_big_df_format(df, name_attempt, path, dictionary_info):
    # dataframe from output network is -> columns = selection, dialect, precision, recall , f1 score, support
    # makes it columns = precision, recall, f1 score, support with rows selection, accuracy, macro avg, weighted avg, classifiers
    # with selection = phone selection, pastes all phone selection together in this format, even if empty
    # also makes a new .txt file with condensed information -> this is in dict_avg
    info_list = list(df.iloc[:,0].dropna().unique())
    df_first = turn_df(df, info_list[0])
    df_empty = make_empty_slot(df_first, "empty")
    df_first = pd.DataFrame(index = list(df_first.index))
    dict_avg = {}
    for info in all_info_list:
        if not info in info_list:
            df_new = make_empty_slot(df_empty, info)
            sigma = None
            st_sigma = None
            f1_score = None
        else:
            df_new = turn_df(df, info)
            f1_score = float(df_new.loc["weighted avg", "f1-score"])
            size = int(df_new.loc['weighted avg', 'support'])
            accuracy = float(df_new.loc["accuracy", 'f1-score'])
            min_size = int(min(df_new.support[2:]))
            max_size = int(max(df_new.support[4:]))
            num_dialects = len(df_new.index) - 4
            best_p = float(max_size/size)
            chance_random = 1/num_dialects
            accuracy_random_diff = accuracy - chance_random
            sigma = float
            st_sigma = float
            if f1_score > 0 and min_size > 1:
                sigma = math.sqrt((best_p * (1 - best_p)) / max_size)
                st_sigma_worst = accuracy_random_diff/sigma
                st_sigma_f1 = (f1_score-chance_random)/(math.sqrt((f1_score * (1-f1_score))/size))
                print(st_sigma_worst, st_sigma_f1, "\n", st_sigma_f1-st_sigma_worst)
                st_sigma = st_sigma_f1
        dict_avg[info] = [f1_score, sigma, st_sigma]
        df_first = pd.concat([df_first, df_new], axis = 1)
    df = df_first
    num_dialects = len(df.index) - 4
    write_dict(name_attempt, dict_avg, num_dialects, x= 0, name =  'output_avg.txt', mult = 100)
    write_dict(name_attempt, dict_avg, num_dialects, x=1, name='sigma.txt', mult = 100)
    write_dict(name_attempt, dict_avg, num_dialects, x=2, name='st_sigma.txt', mult = False)
    df.to_excel(writer, sheet_name = name_attempt)
    return df, info_list

def df_to_excel(df, info_list, name_attempt, dictionary_info):
    workbook = writer.book
    worksheet = writer.sheets[name_attempt]
    worksheet.set_row(13, 70)
    format1 = workbook.add_format({'num_format': '0.00'})
    for n, i in enumerate(all_info_list):
        if i in info_list:
            name = wdir + "/" + i + "_" + name_attempt + "_con_matrix.png"
            cell_x = 2 + (n * 4)
            worksheet.insert_image(13, cell_x, name, {'x_scale': 0.3, 'y_scale': 0.3})
    for num, i in enumerate(dictionary_info):
        worksheet.write_string(num + 20, 1, i)
        worksheet.write_string(num + 20, 2, dictionary_info[i])

    ### shading:
    shaded = []
    for i in range(1,len(all_info_list), 2):
        for j in range(1+(i*4),5+(i*4)):
            shaded.append(j)
    decimal = list(range(4, 4*(len(all_info_list)+1), 4))

    format1 = workbook.add_format({'bg_color':'#CCFFCC', 'num_format': '0.00'})
    format3 = workbook.add_format({'bg_color': '#CCFFCC', 'num_format': '0'})
    format2 = workbook.add_format({'bg_color':'white', 'num_format': '0.00'})
    format4 = workbook.add_format({'bg_color': 'white', 'num_format': '0'})
    for c in range(4*len(all_info_list)+4):
        if c in shaded:
            if c in decimal:
                worksheet.set_column(c, c, None, format3)
            else:
                worksheet.set_column(c, c, None, format1)
        else:
            if c in decimal:
                worksheet.set_column(c, c, None, format4)
            else:
                worksheet.set_column(c, c, None, format2)


#

def clean_info(name_attempt, path):
    dic_info = {}
    with open(path+'/'+'information.txt') as f:
        for line in f.readlines():
            line = line.strip()
            name, rest = line.split("\t")
            dic_info[name] = rest
    return dic_info

def main():
    attempts.sort()
    for a in attempts:
        df,path = clean_df(name_attempt = a)
        info = clean_info(name_attempt = a, path = path)
        df, info_selection = df_to_big_df_format(df, name_attempt=a, path = path, dictionary_info=info)
        df_to_excel(df, name_attempt=a, info_list= info_selection, dictionary_info = info)
        print(a)
    df = pd.read_csv("output_avg.txt", index_col = 0)
    df.to_excel(writer, sheet_name = "f1 average")
    df = pd.read_csv("sigma.txt", index_col = 0)
    df.to_excel(writer, sheet_name = 'sigma')
    df = pd.read_csv("st_sigma.txt", index_col = 0)
    df.to_excel(writer, sheet_name = 'z-score')
    writer.save()
    return df, info_selection


