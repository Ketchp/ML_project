import numpy as np


def train_baseline_classifier(y_train, y_test):
    prediction = 0 if np.sum(y_train == 0) >= np.sum(y_train == 1) else 1

    miss_rate = np.sum(y_test != prediction) / len(y_test)

    return miss_rate, np.full(len(y_test), prediction)
