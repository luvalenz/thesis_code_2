__author__ = 'lucas'

import glob

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import NewDataStructures


def get_whole_macho(training_set_path, data_set_path, offset = None):
    training = pd.read_csv(training_set_path, sep=',', index_col=0)
    X = training.values[:,:-1].astype(np.float32)
    y = training.values[:,-1].astype(np.int32)
    RFC = RandomForestClassifier()
    RFC.fit(X,y)
    importances = RFC.feature_importances_
    features_order = np.argsort(importances)[::-1]
    X = X[:,features_order]
    whole_dataset = np.empty((0, X.shape[1]))
    file_paths = glob.glob("{0}/*".format(data_set_path))
    i = 0
    for file_path in file_paths:
        file_data = pd.read_csv(file_path, sep=',', index_col=0)
        file_data_np = file_data.values.astype(np.float32)
        #print(file_data_np.shape)
        sorted_file_data = file_data_np[:,features_order]
        whole_dataset = np.vstack((whole_dataset, sorted_file_data))
        if offset is not None and offset == i:
            break
        i += 1
    return whole_dataset

if __name__ == '__main__':
    data = get_whole_macho('MACHO_ts2.csv', '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/MACHO_features_Harmonics')
    data = data[:,:5]
    target_indices = np.random.choice(len(data), 100, replace=False)

    our_method = NewDataStructures.OurMethod(data, '0.3', similarity_metric='euclidean')