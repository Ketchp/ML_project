import torch
from torchmetrics.classification import BinaryAccuracy


# export PYTHONPATH="/home/stefan/DTU/ML/02450Toolbox_Python/Tools"
# may be needed before
from toolbox_02450 import train_neural_net


X_train = torch.Tensor(X_outer_train[train_index, :])
y_train = torch.Tensor(np.reshape(y_outer_train[train_index], (-1, 1)))
X_test = torch.Tensor(X_outer_train[test_index, :])
y_test = torch.Tensor(np.reshape(y_outer_train[test_index], (-1, 1)))

for j, hidden_units in enumerate(hidden_units_list):
    print('|', end='', flush=True)


    # Train ANN with hidden_units and save error values
    # test model on test values and save error
    def model():
        return torch.nn.Sequential(
            torch.nn.Linear(M, hidden_units),  # M features to H hidden units
            # 1st transfer function, either Tanh or ReLU:
            torch.nn.Tanh(),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, 1),  # H hidden units to 1 output neuron
            torch.nn.Sigmoid()  # output transfer
        )


    net, floss, learning_curve = train_neural_net(model,
                                                  loss_fn,
                                                  X=X_train,
                                                  y=y_train,
                                                  n_replicates=3,  # increase when everything works
                                                  max_iter=max_iter,
                                                  silent=True)

    accuracy = metric(net(X_test), y_test)

    saved_values['ann'][i][k].append({'model': net,
                                      'error': 1 - accuracy,
                                      'final_loss': floss,
                                      'learning_curve': learning_curve})
