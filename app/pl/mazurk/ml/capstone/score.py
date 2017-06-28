import math

import numpy as np
from sklearn.metrics import mean_squared_error


def default_error_score(test, predictions):
    return math.sqrt(mean_squared_error(test, predictions))


def calculate_baseline(dataset: np.ndarray, evalfunc=default_error_score):
    # Train / Test split
    # =====================================
    train_test_ratio = 0.8
    train_size = int(len(dataset) * train_test_ratio)
    train = dataset[0:train_size]  # type: np.ndarray
    test = dataset[train_size:len(dataset)]  # type: np.ndarray

    # Baseline score (next = prev)
    # =====================================
    history = [x for x in train]  # type: list
    predictions = np.repeat(np.nan, len(test))  # type: np.ndarray
    for i in range(len(test)):
        predictions[i] = history[-1]
        history.append(test[i])
    # report performance
    return default_error_score(test, predictions)
