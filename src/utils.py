from typing import Dict

import numpy as np


def get_class_weights(labels: np.array) -> Dict:
    # See: https://stackoverflow.com/questions/57021620/how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch
    counts = np.bincount(labels)
    weight_for_0 = counts[1]/counts[0]
    weight_for_1 = counts[0]/counts[1]
    class_wts = {0: weight_for_0, 1: weight_for_1}

    return class_wts
