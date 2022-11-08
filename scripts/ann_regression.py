import torch
from torch.nn import MSELoss
from toolbox_02450 import train_neural_net
import numpy as np


loss_fn = MSELoss()


def create_model(input_layer_units, hidden_units):
    return lambda: torch.nn.Sequential(
            torch.nn.Linear(input_layer_units, hidden_units),  # M features to H hidden units
            # 1st transfer function, either Tanh or ReLU:
            torch.nn.Tanh(),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, 1)  # H hidden units to 1 output neuron
        )


def train_ann_regressor_ensemble(X_train, y_train,
                                 X_test, y_test,
                                 units_list, input_layer_units):
    errors = np.ndarray((len(units_list),))

    for j, hidden_units in enumerate(units_list):
        error, _ = train_ann_regressor(X_train, y_train,
                                       X_test, y_test,
                                       hidden_units, input_layer_units)

        errors[j] = error
    return errors


def train_ann_regressor(X_train, y_train,
                        X_test, y_test,
                        hidden_units, input_layer_units):
    model = create_model(input_layer_units, hidden_units)

    net, floss, learning_curve = train_neural_net(model,
                                                  loss_fn,
                                                  X=X_train,
                                                  y=y_train,
                                                  n_replicates=1,  # when everything work increase to 3
                                                  max_iter=5000)   # preferably increase to at least 10000

    y_pred = net(X_test)

    error = loss_fn(y_pred, y_test)

    return error, y_pred
