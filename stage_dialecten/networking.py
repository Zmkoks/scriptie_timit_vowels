import warnings

import keras
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Reshape, Flatten, Conv2D, MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from information import dic_info


possible_activation = ["relu", "softmax", "sigmoid"]


def make_standard_size(length):
    layers_size = []
    for n in range(length, 0, -1):
        layers_size.append(2 ** (n + 3))
    return layers_size


def set_layers(ntype):
    if type(ntype) is list:
        if len(ntype) == 2:
            layers_var = ntype[0]
            layers_size = ntype[1]
        else:
            print('incorrect network type, automatically 2 relu layers made!')
            layers_var = None
            layers_size = None
    else:
        print('incorrect network type, automatically 2 relu layers made!')
        layers_var = None
        layers_size = None

    if layers_var is None:
        layers_var = ["relu", "relu"]
    if layers_size is None:
        layers_size = make_standard_size(len(layers_var))
    if len(layers_var) is not len(layers_size):
        print('Error, length of layers is not equal to layers size array, will do standard size')
        layers_size = make_standard_size(len(layers_var))
    return layers_var, layers_size

def prepare_network(dataset, del_dia, del_sex, save_name, classes, ntype):
    #delete_rows:
    if del_dia:
        dataset = dataset[~dataset[0].isin(del_dia)]

    if del_sex:
        dataset = dataset[~dataset[1].isin(del_sex)]

    # filter by sentence type (SX/SA/SI) -> https://catalog.ldc.upenn.edu/docs/LDC93S1/timit.readme.html
    # dataset = dataset.loc[dataset[3]=='SA']
    # get the class weights by dialect:
    dict_weight = {}
    classes_counts = dataset[classes].value_counts().sort_index()
    dialect_max = classes_counts.max()
    for i, dialect in enumerate(classes_counts):
        dict_weight[i] = dialect_max / dialect

    y = dataset.astype(str).iloc[:, classes].values
    if 'wide' in ntype:
        X = dataset.iloc[:, 6:].values
    else:
        X = dataset.iloc[:, 7:].values

    y = pd.get_dummies(y)

    from sklearn.preprocessing import StandardScaler

    X_train = X

    # Normalize X_train
    mean_val = X_train.mean(axis=0)
    std_val = X_train.std(axis=0)
    X_train = ((X_train - mean_val) / std_val) / 10
    np.savetxt(save_name + "_values.txt", [mean_val, std_val], delimiter=",", fmt="%s")

    y_train = y
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    return(dict_weight, X_train, y_train, input_dim, output_dim)


def make_standard_model(input_dim, output_dim, ntype):
    layers_var, layers_size = set_layers(ntype)
    model = Sequential()
    model.add(Dense(layers_size[0], input_dim=input_dim, activation=layers_var[0]))

    for i in range(1, len(layers_var)):
         model.add(Dense(layers_size[i], activation=layers_var[i]))

    model.add(Dense(output_dim, activation="softmax"))
    return(model)

def make_cnn_model(input_dim, output_dim):
    model = Sequential()
    model.add(Reshape((input_dim,1), input_shape=(input_dim,)))
    model.add(Conv1D(8, 3, activation='relu'))
    model.add(Conv1D(8, 3, activation='relu'))
    model.add(Conv1D(8, 3, activation='relu'))
    model.add(Conv1D(8, 3, activation='relu'))
    model.add(Reshape((8,-1)))
    model.add(MaxPooling1D(1,8))
    model.add(Flatten())
    model.add(Dense(10, activation = 'relu'))


    model.add(Dense(output_dim, activation="softmax"))
    return(model)

def make_cnn_wide_model(x_data, input_dim, output_dim, size, sel):
    selection_size = sel[0] + 1 + sel[1]
    x_data = np.resize(x_data, (len(x_data), selection_size, size))
    print(x_data.shape)

    model = Sequential()
    model.add(Conv2D(8, (1,3), activation='relu', input_shape = (selection_size, size, 1)))
    model.add(Conv2D(8, (int(selection_size/3), int(size/3)), activation='relu'))
    model.add(Conv2D(8, (2, int(size/3)), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten(input_shape=(selection_size, size, 1)))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(output_dim, activation="softmax"))

    return x_data, model

def fit_model(model, X_train, y_train, batchsize, epoch, dict_weight, save_name):
    sgd = keras.optimizers.SGD(learning_rate=5e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    ACCURACY_THRESHOLD = .95
    print(y_train)

    class myCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > ACCURACY_THRESHOLD):
                print("\nReached %2.2f%% accuracy, so stopping training!!" % (ACCURACY_THRESHOLD * 100))
                self.model.stop_training = True

    callbacks = myCallback()
    if not dic_info['weighted']:
        stats = model.fit(X_train, y_train,
                          shuffle=True,
                          batch_size=batchsize,
                          epochs=epoch,
                          verbose=1,
                          callbacks=[callbacks],
                          )
    else:
        stats = model.fit(X_train, y_train,
                          shuffle=True,
                          batch_size=batchsize,
                          epochs=epoch,
                          verbose=1,
                          callbacks=[callbacks],
                          class_weight = dict_weight
                          )

    model.save(save_name)


def run_networking(dataset, save_name="2l_relu_bs128_ep500", del_dia=None, del_sex=None,
                   epoch=500, batchsize=125, classifier_col = 0, ntype ='nn', n_cols = 13,
                   sel = [1,1]):

   dw, Xtr, ytr, input_dim, output_dim= prepare_network(dataset, del_dia, del_sex, save_name, classifier_col, ntype)

   if ntype == 'cnn':
        model = make_cnn_model(input_dim, output_dim)
   elif ntype == 'cnn_wide':
        Xtr, model = make_cnn_wide_model(Xtr, input_dim, output_dim, n_cols, sel)
   else:
        model = make_standard_model(input_dim, output_dim, ntype = ntype)
   fit_model(model=model, X_train=Xtr,y_train = ytr, batchsize = batchsize, epoch = epoch, dict_weight=dw,
              save_name=save_name)

