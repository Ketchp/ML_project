from data_preparation import df
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt

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

# First run wide scan
# lambda_list = np.logspace(start=1, stop=8, num=20)

# Precise tight scan
lambda_list = np.logspace(start=1, stop=4, num=31)

lin_reg_err = np.ndarray((K, len(lambda_list)))
coefficients = np.ndarray((K, len(lambda_list), M))

for i, (train_index, test_index) in enumerate(selector.split(X)):
    X_train, y_train = X[train_index, :], y[train_index]
    X_test, y_test = X[test_index, :], y[test_index]

    for j, reg_lambda in enumerate(lambda_list):
        lm = Ridge(alpha=reg_lambda).fit(X_train, y_train)
        
        y_pred = lm.predict(X_test)
        
        error = np.mean(np.power(y_pred - y_test, 2))

        lin_reg_err[i][j] = error
        coefficients[i][j] = lm.coef_

    print(lambda_list[np.argmin(lin_reg_err[i])])

plt.figure()

for i in range(K):
    plt.plot(lambda_list, lin_reg_err[i])

plt.xscale('log')
plt.xlabel('$\\lambda$')
plt.ylabel('Error')

plt.title('Test model error for different folds.')

# plt.savefig('Regression_A_Error_vs_lambda.png')
plt.savefig('Regression_A_Error_vs_lambda_close.png')
plt.show()


coefficients = np.mean(coefficients, axis=0)


# plt.figure()
#
# # 8 most significant constants
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
# plt.savefig('Regression_A_Weight_vs_lambda.png')
# plt.show()


lin_reg_err = np.mean(lin_reg_err, axis=0)

print(lin_reg_err)

print(lambda_list)
print(lambda_list[np.argmin(lin_reg_err)])

# Ideal lambda per fold 1264.9, 1048.1, 2223.0, 1264.9, 1.0, 1.0, 1.0, 596.4, 1.0, 1.0
# Ideal lambda for mean error 91.0 with error 106.7
