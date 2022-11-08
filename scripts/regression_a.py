from data_preparation import df
from sklearn.model_selection import KFold
import numpy as np

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


K = 10

selector = KFold(K)

lambda_list = np.logspace(start=-5, stop=5, num=10)
lin_reg_err = []

for i, (train_index, test_index) in enumerate(selector.split(X)):
    X_train, y_train = X[train_index, :], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]

    for j, reg_lambda in enumerate(lambda_list):
        pass
        # Train linear regressor with regularisation reg_lambda
        # test model on test values and save error

    # Find ideal lambda
    # Train new models with lambda on X_train / or save model for every lambda and use saved model
    # Compute Etest for table on X_test, y_test
    # copy from classification increase_confusion...
