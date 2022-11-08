import numpy as np


def train_baseline_regressor(y_train, y_test):
    prediction = np.mean(y_train)

    error = np.mean(np.power(y_test - prediction, 2))

    return error, np.full(len(y_test), prediction)
