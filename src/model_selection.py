import numpy as np
from sklearn.model_selection import train_test_split, KFold


def patient_train_test_split(X: np.array, y: np.array, pt_ids: np.array, test_size=None, random_state=None, shuffle=True):
    uniq_pt_ids = np.unique(pt_ids)
    train_uniq_pt_ids, test_uniq_pt_ids = train_test_split(
        uniq_pt_ids, test_size=test_size, random_state=random_state, shuffle=shuffle)
    train_index = np.isin(pt_ids, train_uniq_pt_ids)
    test_index = np.isin(pt_ids, test_uniq_pt_ids)

    return X[train_index], X[test_index], y[train_index], y[test_index]


class PatientKFold():
    def __init__(self, **kwargs):
        self.kf = KFold(**kwargs)

    def split(self, pt_ids: np.array):
        uniq_pt_ids = np.unique(pt_ids)
        for train_index, val_index in self.kf.split(uniq_pt_ids):
            train_bool_index = np.isin(pt_ids, (uniq_pt_ids[train_index]))
            train_index = np.where(train_bool_index)[0]
            test_index = np.where(~train_bool_index)[0]

            yield train_index, test_index
