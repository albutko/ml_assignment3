import numpy as np
class Dataset:
    def __init__(self):
        self.train_data = {}
        self.test_data = {}
        self.classes = []
        self.name = ''

    def _load(self):
        pass

    def get_train_data(self):
        return (self.train_data.get('features').copy(), self.train_data.get('labels').copy())

    def get_test_data(self):
        return (self.test_data.get('features').copy(), self.test_data.get('labels').copy())

    def get_classes(self):
        return self.classes

    def _send_to_txt(self, df, path, fmt='%.18f'):
        np.savetxt(path, df, fmt=fmt, delimiter=',')
