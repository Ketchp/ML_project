from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def train_knn_regressor_ensemble(X_train, y_train,
                                 X_test, y_test, neighbors_list):
    miss_rates = np.ndarray((len(neighbors_list),))
    for j, k_neighbors in enumerate(neighbors_list):
        # Train linear regressor with regularisation reg_lambda
        # test model on test values and save error
        miss_rate, _ = train_knn_regressor(X_train, y_train,
                                           X_test, y_test,
                                           k_neighbors)
        miss_rates[j] = miss_rate

    return miss_rates


def train_knn_regressor(X_train, y_train,
                        X_test, y_test, k_neighbors):
    # data are centered => fit_intercept=False
    knn_model = KNeighborsClassifier(n_neighbors=k_neighbors).fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    miss_rate = np.sum(y_test != y_pred) / len(y_test)  # model miss-classification rate

    return miss_rate, y_pred == y_test
