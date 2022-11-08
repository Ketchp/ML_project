from data_preparation import X, y, N, M
from sklearn.model_selection import KFold
import numpy as np
from baseline_classification import train_baseline_regressor
from linear_classification import train_logistic_regressor_ensemble, train_logistic_regressor
from knn_classification import train_knn_regressor_ensemble, train_knn_regressor
import matplotlib.pyplot as plt
from toolbox_02450 import mcnemar

# center and standardise data
X_offset = np.ones((N, 1)) * X.mean(axis=0)
X = X - X_offset

X_scale = np.std(X, 0)
X *= 1 / X_scale


K_1 = 10
K_2 = 10  # when everything works 10


lambda_list = np.logspace(start=-2, stop=1, num=11)
neighbors_list = np.array((1, 5, 10, 16, 19, 22, 25, 28, 31, 34, 38, 42, 50, 100))


sample_sizes = {key: np.ndarray((K_1, K_2)) for key in ['train', 'test']}

outer_fold_errors = {'lin': np.ndarray((K_1, len(lambda_list))),
                     'knn': np.ndarray((K_1, len(neighbors_list)))}

best_lambdas = np.ndarray((K_1,))
best_k_neighbors = np.ndarray((K_1,))

test_miss_rates = {key: np.ndarray((K_1,)) for key in ('base', 'lin', 'knn')}

predictions = {key: [] for key in ('true', 'base', 'lin', 'knn')}


outer_selector = KFold(K_1)
for i, (outer_train_index, validation_index) in enumerate(outer_selector.split(X)):
    X_outer_train, y_outer_train = X[outer_train_index, :], y[outer_train_index]
    X_validation, y_validation = X[validation_index, :], y[validation_index]

    lin_inner_fold_errors = np.ndarray((K_2, len(lambda_list)))
    inner_fold_coefficients = np.ndarray((K_2, len(lambda_list), M))

    knn_inner_fold_errors = np.ndarray((K_2, len(neighbors_list)))

    inner_selector = KFold(K_2)
    for k, (train_index, test_index) in enumerate(inner_selector.split(X_outer_train)):
        print(k, end=': ', flush=True)
        X_train, y_train = X_outer_train[train_index, :], y_outer_train[train_index]
        X_test, y_test = X_outer_train[test_index, :], y_outer_train[test_index]

        sample_sizes['train'][i][k] = len(y_train)
        sample_sizes['test'][i][k] = len(y_test)

        errors, coefficients = train_logistic_regressor_ensemble(X_train,
                                                                 y_train,
                                                                 X_test,
                                                                 y_test,
                                                                 lambda_list)
        lin_inner_fold_errors[k] = errors
        inner_fold_coefficients[k] = coefficients

        print('|', end='', flush=True)

        errors = train_knn_regressor_ensemble(X_train,
                                              y_train,
                                              X_test,
                                              y_test,
                                              neighbors_list)
        knn_inner_fold_errors[k] = errors
    print()

    # for each lambda/h.u. compute E_gen_s = sum over j = K_2 {|Djtest| * Ej_s / |D_outer_train|}

    lin_inner_fold_errors = (lin_inner_fold_errors.T * sample_sizes['test'][i]).T / sum(sample_sizes['test'][i])
    knn_inner_fold_errors = (knn_inner_fold_errors.T * sample_sizes['test'][i]).T / sum(sample_sizes['test'][i])

    E_gen_lin = np.sum(lin_inner_fold_errors, axis=0)
    E_gen_knn = np.sum(knn_inner_fold_errors, axis=0)

    # Find ideal lambda and hidden units
    best_lambda = lambda_list[np.argmin(E_gen_lin)]
    best_k_neigh = neighbors_list[np.argmin(E_gen_knn)]

    base_miss_rate, base_predictions = train_baseline_regressor(y_outer_train,
                                                                y_validation)

    lin_miss_rate, _, lin_predictions = train_logistic_regressor(X_outer_train,
                                                                 y_outer_train,
                                                                 X_validation,
                                                                 y_validation,
                                                                 best_lambda)

    knn_miss_rate, knn_predictions = train_knn_regressor(X_outer_train,
                                                         y_outer_train,
                                                         X_validation,
                                                         y_validation,
                                                         best_k_neigh)

    outer_fold_errors['lin'][i] = E_gen_lin
    outer_fold_errors['knn'][i] = E_gen_knn

    best_lambdas[i] = best_lambda
    best_k_neighbors[i] = best_k_neigh

    test_miss_rates['base'][i] = base_miss_rate
    test_miss_rates['lin'][i] = lin_miss_rate
    test_miss_rates['knn'][i] = knn_miss_rate

    predictions['true'].append(y_validation)
    predictions['base'].append(base_predictions)
    predictions['lin'].append(lin_predictions)
    predictions['knn'].append(knn_predictions)

    print(f'{i}:', end='\t')
    print(f'{base_miss_rate:.2f}', end='\t')
    print(f'{best_lambda:.2f}: {lin_miss_rate:.2f}', end='\t')
    print(f'{best_k_neigh:.2f}: {knn_miss_rate:.2f}')


# plt.figure()
#
# leg_idx = [0,  1,  3,  4, 15, 23, 34, 35]
# handles = [plt.plot(lambda_list, coefficients[:, i])[0] for i in range(M)]
#
# used = [handles[i] for i in leg_idx]
# names = ['Age',
#          'Education number',
#          'Capital gain',
#          'Hours per week',
#          'Marital status: Married civ.',
#          'Occupation: Exec./Managerial',
#          'Relationship: Married',
#          'Relationship: Not in family']
# plt.legend(used, names)
#
# plt.xscale('log')
#
# plt.xlabel('$\\lambda$')
# plt.ylabel('$w_i$')
#
# plt.title('Parameter weights')
#
# plt.savefig('LogisticRegressorWeight_vs_lambda.png')
# plt.show()


# Plot logistic regressor error
plt.figure()

# plt.plot(lambda_list, np.sum(outer_fold_errors['lin'], axis=0))
for i in range(K_1):
    plt.plot(lambda_list, outer_fold_errors['lin'][i])

plt.xscale('log')

plt.xlabel('$\\lambda$')
plt.ylabel('Error')

plt.title('Test model error for different folds.')

# plt.savefig('LogisticRegressorError_vs_lambda_close.png')
plt.show()

# Plot KNN classifier error
plt.figure()

# plt.plot(neighbors_list, np.mean(outer_fold_errors['knn'], axis=0))
for i in range(K_1):
    plt.plot(neighbors_list, outer_fold_errors['knn'][i])


plt.xlabel('K neighbors')
plt.ylabel('Error')

# plt.savefig('KNNClassifierError_vs_K.png')
plt.show()

# table for report
print('Fold\t', 'lambda', 'Error', 'K', 'Error', 'Error', sep='\t')
for i in range(K_1):
    print(i, '\t',
          f'{best_lambdas[i]:.2f}',
          f'{test_miss_rates["lin"][i]:.2f}',
          int(best_k_neighbors[i]),
          f'{test_miss_rates["knn"][i]:.2f}',
          f'{test_miss_rates["base"][i]:.2f}', sep='\t')

# Statistics evaluation
for model in predictions:
    predictions[model] = np.concatenate(predictions[model])

print("Baseline vs logistic regressor model:")
mcnemar(predictions['true'],
        predictions['base'],
        predictions['lin'])

print("Baseline vs KNN classifier model:")
mcnemar(predictions['true'],
        predictions['base'],
        predictions['knn'])

print("Logistic regressor vs KNN classifier model:")
mcnemar(predictions['true'],
        predictions['lin'],
        predictions['knn'])


# save values if needed for later
# with open('file_name.pickle', 'wb') as file:
#     import pickle
#     pickle.dump({'Anything you want': "Nothing"}, file)
