import keras
import pandas as pd
import numpy as np
import sklearn.metrics
from colorspacious import cspace_converter
from information import dic_info

#groups_inverse = {1: 2, 2: 2, 3: 2, 6: 5, 4: 5, 5: 5, 7: 'rest', 8 : 'rest'}

def run_output(model_name, dataset, del_dia, del_sex, classes = 0, size = 13, sel = [1,1]):

    model = keras.models.load_model(model_name)

    #dataset = dataset.replace({0:groups_inverse})
    if del_dia:
        dataset = dataset[~dataset[0].isin(del_dia)]
    if del_sex:
        dataset = dataset[~dataset[1].isin(del_sex)]

    #filter by sentence type (SX/SA/SI) -> https://catalog.ldc.upenn.edu/docs/LDC93S1/timit.readme.html
    #dataset = dataset.loc[dataset[3]=='SA']
    y_test = dataset.astype(str).iloc[:, classes].values
    labels = np.unique(dataset.astype(str).iloc[:,classes]).tolist()
    if 'wide' in dic_info['type']:
        X_test = dataset.iloc[:, 6:].values
    else:
        X_test = dataset.iloc[:, 7:].values
    y_test = pd.get_dummies(y_test)

    #Normalizing the data
    from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_test = sc.fit_transform(X_test)
    values = pd.read_csv(model_name+"_values.txt", header = None).values.squeeze()
    mean_val = values[0]
    std_val = values[1]
    X_test = ((X_test - mean_val) / std_val)/10

    if 'wide' in  dic_info['type']:
        selection_size = sel[0] + 1 + sel[1]
        X_test = np.resize(X_test, (len(X_test), selection_size, size))

    y_pred = model.predict(X_test)
    #Converting predictions to label
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))

    #Converting one hot encoded test label to label
    test = np.argmax(y_test.values, axis=1)


    #converting data to string labels:
    data_labels = np.unique(test)
    dict_labels = {}
    for n,i in enumerate(labels):
        dict_labels[data_labels[n]] = i
    test = pd.Series(test).replace(dict_labels)
    pred = pd.Series(pred).replace(dict_labels)
    print(dict_labels)


    from sklearn.metrics import accuracy_score
    a = accuracy_score(pred,test)
    print(model_name + '\t :Accuracy is:', a*100)
    from sklearn.metrics import classification_report
    from sklearn.metrics import matthews_corrcoef
    labels = labels
    c = classification_report(test,pred, target_names = labels, output_dict = True, zero_division = 0 )
    m = matthews_corrcoef(test, pred)
    from sklearn.metrics import confusion_matrix
    con = confusion_matrix(test, pred, normalize = 'true', labels = labels)
    x = pd.DataFrame(con, columns = labels, index = labels)
    x.to_csv(model_name+"_con_matrix"+".txt")

    return c,m