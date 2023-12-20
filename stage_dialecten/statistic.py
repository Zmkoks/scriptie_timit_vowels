import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

filename ='/media/zjoske/Seagate Expansion Drive/scriptie/13_160_400/all_vowels'
groups = {'north': [1, 2, 3, 6], 'south': [4, 5], 'west': [7]}
groups_inverse = {1: 'north', 2: 'north', 3: 'north', 6: 'north', 4: 'south', 5: 'south', 7: 'rest', 8 : 'rest'}
vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix',
          'axr', 'ax-h']
all_info = {'all_vowels': vowels,
            'all_i': ['ih', 'ix', 'iy'],
            'all_u': ['uh', 'uw', 'ux'],
            'all_a': ['aa', 'ae', 'ah'], 'all_mid': ['ax-h', 'ax', 'axr', 'eh', 'ay', 'er', 'ow'], 'just_i': ['iy'],
            'just_u': ['uw'], 'just_a': ['aa'], 'just_mid': ['ax']
            }

all_info_inverse  = {'iy': 'all_i', 'ih': 'all_i', 'eh': 'all_mid', 'ey': 'all_vowels', 'ae': 'all_a', 'aa': 'all_a', 'aw': 'all_vowels', 'ay': 'all_mid', 'ah': 'all_a', 'ao': 'all_vowels', 'oy': 'all_vowels', 'ow': 'all_mid', 'uh': 'all_u', 'uw': 'all_u', 'ux': 'all_u', 'er': 'all_mid', 'ax': 'all_mid', 'ix': 'all_i', 'axr': 'all_mid', 'ax-h': 'all_mid'}
def preprocessing(filename="all_data_13_25_ms", plot1=0, plot2=1, sel=0, comp=2, dialects=None, gender=None,
                  dic_group=None, vowels_group = None):
    filename1 = filename + '_TEST.txt'
    filename2 = filename + '_TRAIN.txt'
    dataframe1 = pd.read_csv(filename1, header=None)
    dataframe2 = pd.read_csv(filename2, header=None)
    dataframe = pd.concat([dataframe1, dataframe2])
    if dialects is not None:
        dataframe = dataframe[~dataframe[0].isin(dialects)]
    if dic_group is not None:
        dataframe = dataframe.replace({0: dic_group})
    if gender is not None:
        dataframe = dataframe[~dataframe[1].isin(gender)]


    #dataframe = dataframe.groupby(by=[0,1,2,3,4,5], as_index = False).mean().drop([6])
    dataframe = dataframe.dropna()

    print("shape is: " + str(dataframe.shape))
    dialect_n = dataframe[sel].value_counts(normalize=True) * 100
    print("dialect distribution is: " + str(dialect_n))
    x_df = dataframe.iloc[:, 7:]
    y_df = dataframe.iloc[:, sel].values
    x_stan = StandardScaler().fit_transform(x_df)
    pca = PCA(n_components=comp)
    pca_x = pca.fit_transform(x_stan)
    pca_df = pd.DataFrame(data=pca_x, columns=list(range(comp)))
    pca_df['y'] = y_df
    if vowels_group is not None:
        pca_df = pca_df.replace({'y':vowels_group})
        dialect_n = pca_df['y'].value_counts(normalize=True) * 100
    return pca_df
    plt.figure(figsize=(16, 6))
    sns.scatterplot(
        x=plot1, y=plot2,
        hue='y',
        palette=sns.color_palette('pastel', len(dialect_n)),
        data=pca_df,
        alpha=0.3
    )
    plt.show()
    return (pca_df)
