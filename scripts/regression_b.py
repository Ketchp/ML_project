from data_preparation import df
from sklearn.model_selection import KFold
import numpy as np
import torch
from baseline_regression import train_baseline_regressor
from linear_regression import train_linear_regressor, train_linear_regressor_ensemble
from ann_regression import train_ann_regressor, train_ann_regressor_ensemble
import scipy.stats as st


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

K_1 = 10
K_2 = 10


lambda_list = np.logspace(start=1, stop=3, num=11)
hidden_units_list = np.array((1, 2, 4, 6, 8, 10, 12, 15))


outer_fold_errors = {'lin': np.ndarray((K_1, len(lambda_list))),
                     'ann': np.ndarray((K_1, len(hidden_units_list)))}

best_lambdas = np.ndarray((K_1,))
best_hid_units = np.ndarray((K_1,))

test_miss_rates = {key: np.ndarray((K_1,)) for key in ('base', 'lin', 'ann')}

predictions = {key: [] for key in ('true', 'base', 'lin', 'ann')}

use_folds = range(4, 4)
# use_folds = range(4, 7)
# use_folds = range(7, 10)

outer_selector = KFold(K_1)
for i, (outer_train_index, validation_index) in enumerate(outer_selector.split(X)):
    if i not in use_folds:
        continue

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

    predictions['true'].append(y_validation)
    predictions['base'].append(base_predictions)
    predictions['lin'].append(lin_predictions)
    predictions['ann'].append(ann_predictions)

    print(f'{i}:', end='\t')
    print(f'{base_miss_rate}', end='\t')
    print(f'{best_lambda:.2f}: {lin_miss_rate:.2f}', end='\t')
    print(f'{best_hidden_unit:.2f}: {ann_miss_rate:.2f}', end='')
    print()


# with open(f'fold_{use_folds[0]}-{use_folds[-1]}.pickle', 'wb') as file:
#     import pickle
#     pickle.dump({'outer_fold_errors': outer_fold_errors,
#                  'best_lambdas': best_lambdas,
#                  'best_hid_units': best_hid_units,
#                  'test_miss_rates': test_miss_rates,
#                  'predictions': predictions,
#                  'lin_inner_fold_errors': lin_inner_fold_errors,
#                  'inner_fold_coefficients': inner_fold_coefficients,
#                  'ann_inner_fold_errors': ann_inner_fold_errors,
#                  'used_folds': use_folds}, file)


# table for report
print('Fold\t', 'lambda', 'Error', 'units', 'Error', 'Error', sep='\t')
for i in range(K_1):
    print(i, '\t',
          f'{best_lambdas[i]:.4f}',
          f'{test_miss_rates["lin"][i]:.1f}',
          int(best_hid_units[i]),
          f'\t{test_miss_rates["ann"][i]:.1f}',
          f'{test_miss_rates["base"][i]:.1f}', sep='\t')

# Table:      Lin. model      ANN model    baseline
# Fold		lambda	Error	units	Error	Error
# 0			63.0957	106.8	8	    101.0	170.7
# 1			63.0957	108.9	6	    101.9	172.2
# 2			39.8107	110.4	12	    100.8	174.7
# 3			63.0957	107.8	6	    102.5	176.6
# 4			63.0957	105.5	6	    101.3	177.4
# 5			63.0957	101.1	8	    93.8	167.9
# 6			63.0957	101.4	10	    93.3	168.7
# 7			39.8107	105.3	6	    97.6	176.7
# 8			63.0957	109.8	10	    100.8	182.1
# 9			63.0957	109.9	10	    100.9	180.3


# plt.figure()
#
# for i in range(K_1):
#     plt.plot(lambda_list, outer_fold_errors['lin'][i])
#
# plt.xscale('log')
# plt.xlabel('$\\lambda$')
# plt.ylabel('Error')
#
# plt.title('Test model error for different folds.')
#
# plt.savefig('RegressionError_vs_lambda.png')
# plt.show()

# plt.figure()
#
# leg_idx = [5,  8, 10, 13, 17, 19, 34, 37]
# handles = [plt.plot(lambda_list, coefficients[:, i])[0] for i in range(M)]
#
# used = [handles[i] for i in leg_idx]
# names = ['From US',
#          'Work-class: Private',
#          'Work-class: Self emp. not inc.',
#          'Marital status: Divorced',
#          'Marital status: Never married',
#          'Marital status: Widowed',
#          'Relationship: Married',
#          'Relationship: Own child']
# plt.legend(used, names)
#
# plt.xscale('log')
#
# plt.xlabel('$\\lambda$')
# plt.ylabel('$w_i$')
#
# plt.title('Parameter weights')
#
# plt.savefig('RegressionWeight_vs_lambda.png')
# plt.show()
#
# plt.figure()
#
# plt.plot(lambda_list[:-2], np.mean(outer_fold_errors['lin'][:, :-2], axis=0))
#
# plt.xscale('log')
# plt.xlabel('$\\lambda$')
# plt.ylabel('Error')
#
# plt.title('Mean model error from all folds.')
#
# plt.savefig('Regression_B_MeanError_vs_lambda_close.png')
# plt.show()
#
# plt.figure()
#
# for row in range(K_1):
#     plt.plot(lambda_list, outer_fold_errors['lin'][row])
#
# plt.xscale('log')
# plt.xlabel('$\\lambda$')
# plt.ylabel('Error')
#
# plt.title('Model error for different folds.')
#
# plt.savefig('Regression_B_Error_vs_lambda.png')
# plt.show()
#
# plt.figure()
#
# plt.plot(hidden_units_list, np.mean(outer_fold_errors['ann'], axis=0))
#
# plt.xlabel('no. hidden nodes')
# plt.ylabel('Error')
#
# plt.title('Mean model error from all folds.')
#
# plt.savefig('Regression_B_MeanError_vs_units.png')
# plt.show()
#
# plt.figure()
#
# plt.plot(hidden_units_list[1:], np.mean(outer_fold_errors['ann'], axis=0)[1:])
#
# plt.xlabel('no. hidden nodes')
# plt.ylabel('Error')
#
# plt.title('Mean model error from all folds.')
#
# plt.savefig('Regression_B_MeanError_vs_units_close.png')
# plt.show()
#
# plt.figure()
#
# for row in range(K_1):
#     plt.plot(hidden_units_list, outer_fold_errors['ann'][row])
#
# plt.xlabel('no. hidden nodes')
# plt.ylabel('Error')
#
# plt.title('Model error for different folds.')
#
# plt.savefig('Regression_B_Error_vs_units.png')
# plt.show()


for model in predictions:
    try:
        predictions[model] = np.concatenate(predictions[model])
    except RuntimeError:
        predictions[model] = np.reshape(np.concatenate([p.detach().numpy() for p in predictions[model]]), (-1))


z = {key: np.power(predictions[key] - predictions['true'], 2) for key in ['base',
                                                                          'lin',
                                                                          'ann']}

alpha = 0.05
CI = {key: st.t.interval(1-alpha,
                         df=z[key].size-1,
                         loc=z[key].mean(),
                         scale=st.sem(z[key])) for key in ['base', 'lin', 'ann']}

for key, value in CI.items():
    print(f'{key} {100*(1-alpha)}% CI: {value}')


# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
pairs = (('base', 'lin'),
         ('base', 'ann'),
         ('lin', 'ann'))
for pair in pairs:
    a, b = pair
    print(f'{a} vs {b} model:')
    z_diff = z[a] - z[b]
    CI_diff = st.t.interval(1-alpha,
                            z_diff.size - 1,
                            loc=z_diff.mean(),
                            scale=st.sem(z_diff))
    print(CI_diff)

    p = 2*st.t.cdf(-np.abs(z_diff.mean())/st.sem(z_diff), df=z_diff.size - 1)  # p-value

    print(f'p-value: {p}')
    print()


# base 95.0% CI: (172.525549490786, 176.91686242143064)
# lin 95.0% CI: (104.99894526646253, 108.42168886572148)
# ann 95.0% CI: (97.71902755497766, 101.06157765021783)
#
# base vs lin model:
# (66.3455143159158, 69.6762634641168)
# p-value: 0.0
#
# base vs ann model:
# (73.45962770604766, 77.20217900097356)
# p-value: 0.0
#
# lin vs ann model:
# (6.635770071446844, 8.004258855541762)
# p-value: 3.742286113231391e-97
