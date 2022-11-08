from sklearn import linear_model
import numpy as np


def train_linear_regressor_ensemble(X_train, y_train,
                                    X_test, y_test,
                                    lambda_list):
    errors = np.ndarray((len(lambda_list),))
    coefficient_lists = np.ndarray((len(lambda_list), X_train.shape[1]))

    for j, reg_lambda in enumerate(lambda_list):

        error, coefficients, _ = train_linear_regressor(X_train, y_train,
                                                        X_test, y_test,
                                                        reg_lambda)

        errors[j] = error
        coefficient_lists[j] = coefficients

    return errors, coefficient_lists


def train_linear_regressor(X_train, y_train,
                           X_test, y_test,
                           reg_lambda):
    # data are centered => fit_int=False
    lm = linear_model.Ridge(alpha=reg_lambda, fit_intercept=False).fit(X_train, y_train)

    y_pred = lm.predict(X_test)

    error = np.mean(np.power(y_test - y_pred, 2))  # model error

    return error, lm.coef_, y_pred
