from data_preparation import df
from sklearn.model_selection import KFold
import numpy as np
import torch
from baseline_regression import train_baseline_regressor
from linear_regression import train_linear_regressor, train_linear_regressor_ensemble
from ann_regression import train_ann_regressor, train_ann_regressor_ensemble

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

K_1 = K_2 = 2  # when everything works, set to 10


lambda_list = np.logspace(start=-5, stop=5, num=5)  # TODO: first try to find suitable ranges
hidden_units_list = range(1, 12, 3)  # TODO: try to find suitable ranges


outer_fold_errors = {'lin': np.ndarray((K_1, len(lambda_list))),
                     'ann': np.ndarray((K_1, len(hidden_units_list)))}

best_lambdas = np.ndarray((K_1,))
best_hid_units = np.ndarray((K_1,))

test_miss_rates = {key: np.ndarray((K_1,)) for key in ('base', 'lin', 'ann')}

predictions = {key: [] for key in ('true', 'base', 'lin', 'ann')}


outer_selector = KFold(K_1)
for i, (outer_train_index, validation_index) in enumerate(outer_selector.split(X)):
    X_outer_train, y_outer_train = X[outer_train_index, :], y[outer_train_index]
    X_validation, y_validation = X[validation_index, :], y[validation_index]

    lin_inner_fold_errors = np.ndarray((K_2, len(lambda_list)))
    inner_fold_coefficients = np.ndarray((K_2, len(lambda_list), M))

    ann_inner_fold_errors = np.ndarray((K_2, len(hidden_units_list)))

    inner_selector = KFold(K_2)
    for k, (train_index, test_index) in enumerate(inner_selector.split(X_outer_train)):
        X_train, y_train = X_outer_train[train_index, :], y_outer_train[train_index]
        X_test, y_test = X_outer_train[test_index, :], y_outer_train[test_index]

        errors, coefficients = train_linear_regressor_ensemble(X_train,
                                                               y_train,
                                                               X_test,
                                                               y_test,
                                                               lambda_list)

        lin_inner_fold_errors[k] = errors
        inner_fold_coefficients[k] = coefficients

        X_train = torch.Tensor(X_outer_train[train_index, :])
        y_train = torch.Tensor(np.reshape(y_outer_train[train_index], (-1, 1)))
        X_test = torch.Tensor(X_outer_train[test_index, :])
        y_test = torch.Tensor(np.reshape(y_outer_train[test_index], (-1, 1)))

        # TODO: increase n_replicates & max_iter when everything works
        errors = train_ann_regressor_ensemble(X_train,
                                              y_train,
                                              X_test,
                                              y_test,
                                              hidden_units_list,
                                              M)

        ann_inner_fold_errors[k] = errors

    E_gen_lin = np.sum(lin_inner_fold_errors, axis=0)
    E_gen_ann = np.sum(ann_inner_fold_errors, axis=0)

    # Find ideal lambda and hidden units
    best_lambda = lambda_list[np.argmin(E_gen_lin)]
    best_hidden_unit = hidden_units_list[np.argmin(E_gen_ann)]
    
    base_miss_rate, base_predictions = train_baseline_regressor(y_outer_train,
                                                                y_validation)

    lin_miss_rate, _, lin_predictions = train_linear_regressor(X_outer_train,
                                                               y_outer_train,
                                                               X_validation,
                                                               y_validation,
                                                               best_lambda)

    ann_miss_rate, ann_predictions = train_ann_regressor(torch.Tensor(X_outer_train),
                                                         torch.Tensor(np.reshape(y_outer_train, (-1, 1))),
                                                         torch.Tensor(X_validation),
                                                         torch.Tensor(np.reshape(y_validation, (-1, 1))),
                                                         best_hidden_unit,
                                                         M)

    outer_fold_errors['lin'][i] = E_gen_lin
    outer_fold_errors['ann'][i] = E_gen_ann

    best_lambdas[i] = best_lambda
    best_hid_units[i] = best_hidden_unit

    test_miss_rates['base'][i] = base_miss_rate
    test_miss_rates['lin'][i] = lin_miss_rate
    test_miss_rates['ann'][i] = ann_miss_rate

    predictions['base'].append(base_predictions)
    predictions['lin'].append(lin_predictions)
    predictions['ann'].append(ann_predictions)

    print(f'{i}:', end='\t')
    print(f'{base_miss_rate}', end='\t')
    print(f'{best_lambda:.2f}: {lin_miss_rate:.2f}', end='\t')
    print(f'{best_hidden_unit:.2f}: {ann_miss_rate:.2f}')


# TODO: print results
# table for report
print('Fold\t', 'lambda', 'Error', 'units', 'Error', 'Error', sep='\t')
for i in range(K_1):
    print(i, '\t',
          f'{best_lambdas[i]:.4f}',
          f'{test_miss_rates["lin"][i]:.1f}',
          int(best_hid_units[i]),
          f'{test_miss_rates["knn"][i]:.1f}',
          f'{test_miss_rates["base"][i]:.1f}', sep='\t')

# TODO: create plots


for model in predictions:
    predictions[model] = np.concatenate(predictions[model])

# TODO: compute statics (e.g. ex7_2_1.py for pairs (baseline, linear), (baseline, ANN), (linear, ANN))
