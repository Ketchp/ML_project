from data_preparation import df
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import torch
from toolbox_02450 import train_neural_net

regression_target = 'age'

attributeNames = df.columns.drop(regression_target)

X = df[attributeNames].to_numpy()
y = df[regression_target].to_numpy()

N, M = X.shape

# center and standardise data
X_offset = np.ones((N, 1)) * X.mean(axis=0)
X = X - X_offset

X_scale = np.std(X, 0)
X *= 1 / X_scale

y_offset = y.mean()
y = y - y_offset

K_1 = K_2 = 2  # when everything works 10


lambda_list = np.logspace(start=-5, stop=5, num=5)
hidden_units_list = range(1, 12, 3)
loss_fn = torch.nn.MSELoss()
max_iter = 5000  # 10000

outer_selector = KFold(K_1)
for i, (outer_train_index, validation_index) in enumerate(outer_selector.split(X)):
    X_outer_train, y_outer_train = X[outer_train_index, :], y[outer_train_index]
    X_validation, y_validation = X[validation_index, :], y[validation_index]

    lin_reg_err = np.ndarray((K_2, len(lambda_list)))
    ann_reg_err = np.ndarray((K_2, len(hidden_units_list)))

    inner_selector = KFold(K_2)
    for k, (train_index, test_index) in enumerate(inner_selector.split(X_outer_train)):
        X_train, y_train = X_outer_train[train_index, :], y_outer_train[train_index]
        X_test, y_test = X_outer_train[test_index, :], y_outer_train[test_index]

        for j, reg_lambda in enumerate(lambda_list):
            # Train linear regressor with regularisation reg_lambda
            # test model on test values and save error

            # data are centered => fit_int=False
            lm = linear_model.Ridge(alpha=reg_lambda, fit_intercept=False).fit(X_train, y_train)

            y_pred = lm.predict(X_test)

            E_k_j = np.mean(np.power(y_test - y_pred, 2))  # model error

            lin_reg_err[k, j] = len(y_test) / len(y_outer_train) * E_k_j

        X_train = torch.Tensor(X_outer_train[train_index, :])
        y_train = torch.Tensor(np.reshape(y_outer_train[train_index], (-1, 1)))
        X_test = torch.Tensor(X_outer_train[test_index, :])
        y_test = torch.Tensor(np.reshape(y_outer_train[test_index], (-1, 1)))

        for j, hidden_units in enumerate(hidden_units_list):
            # Train ANN with hidden_units and save error values
            # test model on test values and save error
            def model():
                return torch.nn.Sequential(
                    torch.nn.Linear(M, hidden_units),  # M features to H hidden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),
                    # torch.nn.ReLU(),
                    torch.nn.Linear(hidden_units, 1)  # H hidden units to 1 output neuron
                )


            net, floss, learning_curve = train_neural_net(model,
                                                          loss_fn,
                                                          X=X_train,
                                                          y=y_train,
                                                          n_replicates=1,  # increase when everything works
                                                          max_iter=max_iter)

            E_k_j = loss_fn(net(X_test), y_test)
            ann_reg_err[k, j] = len(y_test) / len(y_outer_train) * E_k_j

    # for each lambda/h.u. compute E_gen_s = sum over j = K_2 {|Djtest| * Ej_s / |D_outer_train|}
    E_gen_lin = np.sum(lin_reg_err, axis=0)  # maybe axis = 1
    E_gen_ann = np.sum(ann_reg_err, axis=0)  # maybe axis = 1

    # Find ideal lambda and hidden units
    best_lambda = lambda_list[np.argmin(E_gen_lin)]
    best_hidden_units = hidden_units_list[np.argmin(E_gen_ann)]

    # Train new models with the best lambda and hidden units on X_outer_train
    # Compute E_test for table on X_validation
    # compute error for baseline model (e.g. age = avg(age)) and print/save one row of table
