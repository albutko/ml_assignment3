import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from Dataset import Dataset

class MappingDataset(Dataset):

    def __init__(self):
        self.classes = ['water','forest','impervious','farm','grass','orchard']
        self.datadir = os.path.abspath(os.path.dirname(__file__))
        self.name = 'mapping'
        self.test_data = {}
        self.train_data = {}
        self.classes = []
        self._load()


    def _load(self):

        train_data = pd.read_csv(os.path.join(self.datadir,'../../data/mapping/train.csv'))
        test_data = pd.read_csv(os.path.join(self.datadir,'../../data/mapping/test.csv'))

        le = LabelEncoder()
        le.fit(train_data['class'])
        train_data['class'] = le.transform(train_data['class'])
        test_data['class'] = le.transform(test_data['class'])

        self.classes = le.classes_

        X_train = np.array(train_data.iloc[:,1:])
        y_train = np.array(train_data['class'])
        X_test = np.array(test_data.iloc[:,1:])
        y_test = np.array(test_data['class'])

        scaler = StandardScaler().fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        self.train_data = {
            'features': X_train,
            'labels': y_train
        }

        self.test_data = {
            'features': X_test,
            'labels': y_test
        }
