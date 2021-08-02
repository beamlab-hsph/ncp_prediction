
from typing import Tuple, Dict
import pickle
from pathlib import Path
import numpy as np


def _convert_to_arrays(X: Dict, y: Dict, z: Dict) -> Tuple[np.array, np.array, Dict, np.array]:
    pt_list = list(X.keys())
    print(f"Total patients: {len(pt_list)}")

    X_arr = np.vstack([X[pt] for pt in pt_list])
    y_arr = np.hstack([y[pt] for pt in pt_list])
    pt_ids = np.hstack([pt for pt in pt_list for _ in range(X[pt].shape[0])])

    print(f"Total images: {X_arr.shape[0]}")
    print(f"Total positive images: {y_arr.sum()}")

    return X_arr, y_arr, z, pt_ids


def get_train_data(input_dir: Path) -> Tuple[np.array, np.array, Dict, np.array]:
    with open(input_dir/'X.pickle', 'rb') as handle:
        X = pickle.load(handle)
    with open(input_dir/'y.pickle', 'rb') as handle:
        y = pickle.load(handle)
    with open(input_dir/'z.pickle', 'rb') as handle:
        z = pickle.load(handle)

    return _convert_to_arrays(X, y, z)


def get_test_data(input_dir: Path) -> Tuple[np.array, np.array, Dict, np.array]:
    with open(input_dir/'X_test.pickle', 'rb') as handle:
        X = pickle.load(handle)
    with open(input_dir/'y_test.pickle', 'rb') as handle:
        y = pickle.load(handle)
    with open(input_dir/'z_test.pickle', 'rb') as handle:
        z = pickle.load(handle)

    return _convert_to_arrays(X, y, z)
