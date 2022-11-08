from sklearn import linear_model
import numpy as np


def train_logistic_regressor_ensemble(X_train, y_train,
                                      X_test, y_test, lambda_list):
    miss_rates = np.ndarray((len(lambda_list),))
    coefficient_lists = np.ndarray((len(lambda_list), X_train.shape[1]))
    for j, reg_lambda in enumerate(lambda_list):
        # Train linear regressor with regularisation reg_lambda
        # test model on test values and save error
        miss_rate, coefficients, _ = train_logistic_regressor(X_train, y_train,
                                                            X_test, y_test,
                                                            reg_lambda)
        miss_rates[j] = miss_rate
        coefficient_lists[j] = coefficients

    return miss_rates, coefficient_lists


def train_logistic_regressor(X_train, y_train,
                             X_test, y_test, norm_lambda):
    # data are centered => fit_intercept=False
    lm = linear_model.LogisticRegression(fit_intercept=False, C=1 / norm_lambda).fit(X_train, y_train)

    y_pred = lm.predict(X_test)

    miss_rate = np.sum(y_test != y_pred) / len(y_test)  # model miss-classification rate

    return miss_rate, lm.coef_, y_pred == y_test
