from data_preparation import X, y, N, M
from sklearn.model_selection import KFold
import numpy as np
from baseline_classification import train_baseline_regressor
from linear_classification import train_logistic_regressor_ensemble, train_logistic_regressor
from knn_classification import train_knn_regressor_ensemble, train_knn_regressor

# center and standardise data
X_offset = np.ones((N, 1)) * X.mean(axis=0)
X = X - X_offset

X_scale = np.std(X, 0)
X *= 1 / X_scale


K_1 = 10
K_2 = 10  # when everything works 10


lambda_list = np.logspace(start=-4, stop=6, num=11)
neighbors_list = np.array((1, 2, 3, 5, 7, 10, 15, 25, 50, 100, 200, 500))

sample_sizes = {key: np.ndarray((K_1, K_2)) for key in ['train', 'test']}

lin_outer_fold_errors = np.ndarray((K_1, len(lambda_list)))
knn_outer_fold_errors = np.ndarray((K_1, len(neighbors_list)))

best_lambdas = np.ndarray((K_1,))
best_k_neighbors = np.ndarray((K_1,))

base_test_miss_rates = np.ndarray((K_1,))
lin_test_miss_rates = np.ndarray((K_1,))
knn_test_miss_rates = np.ndarray((K_1,))

pair_confusion = {key: np.ndarray((2, 2)) for key in ['base_lin',
                                                      'base_knn',
                                                      'lin_knn']}


def increase_confusion(arr: np.ndarray, first, second):
    arr[0][0] += np.sum(np.logical_and(first, second))
    arr[0][1] += np.sum(np.logical_and(np.logical_not(first),
                                       second))
    arr[1][0] += np.sum(np.logical_and(first,
                                       np.logical_not(second)))
    arr[1][1] += np.sum(np.logical_and(np.logical_not(first),
                                       np.logical_not(second)))


outer_selector = KFold(K_1)
for i, (outer_train_index, validation_index) in enumerate(outer_selector.split(X)):
    print(f'Fold: {i}', flush=True)
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

    base_miss_rate, base_confusion_list = train_baseline_regressor(y_outer_train,
                                                                   y_validation)

    lin_miss_rate, _, lin_confusion_list = train_logistic_regressor(X_outer_train,
                                                                    y_outer_train,
                                                                    X_validation,
                                                                    y_validation,
                                                                    best_lambda)

    knn_miss_rate, knn_confusion_list = train_knn_regressor(X_outer_train,
                                                            y_outer_train,
                                                            X_validation,
                                                            y_validation,
                                                            best_k_neigh)

    lin_outer_fold_errors[i] = E_gen_lin
    knn_outer_fold_errors[i] = E_gen_knn

    best_lambdas[i] = best_lambda
    best_k_neighbors[i] = best_k_neigh

    lin_test_miss_rates[i] = lin_miss_rate
    knn_test_miss_rates[i] = knn_miss_rate

    increase_confusion(pair_confusion['base_lin'], base_confusion_list, lin_confusion_list)
    increase_confusion(pair_confusion['base_knn'], base_confusion_list, knn_confusion_list)
    increase_confusion(pair_confusion['lin_knn'], lin_confusion_list, knn_confusion_list)

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
