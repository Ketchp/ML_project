import numpy as np
import pandas as pd
from scipy.linalg import svd
import matplotlib.pyplot as plt

filename = 'adult.data'
df = pd.read_csv(filename, sep=', ',  header=None, engine='python')

names = pd.read_csv('header.data')
df.columns = names.columns

# drop education( we have ed. number )
df.drop(columns='education', inplace=True)

# (sex, result) to binary
classNames = df['result'].unique()
C = len(classNames)
classDict = dict(zip(classNames, range(C)))

df['result'].replace(classDict, inplace=True)
df['sex'].replace({'Male': 0, 'Female': 1}, inplace=True)

# drop missing values
df.replace({'?': np.nan}, inplace=True)
df.dropna(inplace=True)


def encode_one_of_k(dataframe: pd.DataFrame, column: str | list):
    if type(column) == str:
        from functools import reduce
        return reduce(encode_one_of_k, column, dataframe)

    one_hot = pd.get_dummies(dataframe[column])
    return dataframe.drop(columns=column).join(one_hot)


# (work class, marital status, occupation, relationship, race, native-country) change to 1 of K
nominal_columns = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
df = encode_one_of_k(df, nominal_columns)

attributeNames = df.columns
attributeNames.drop('result')

X = df[attributeNames].to_numpy()
y = df['result'].to_numpy()

N, M = X.shape

print(attributeNames)
print(N, M)
print(X, y.T)
print(C, classNames)


# data normalisation
Y = X - np.ones((N, 1)) * X.mean(axis=0)
Y *= 1 / np.std(Y, 0)

U, S, Vh = svd(Y, full_matrices=False)

rho = S*S / (S*S).sum()

cumulative_rho = np.cumsum(rho)

threshold = 0.9
PC_required = np.argmax(cumulative_rho > threshold)
print(f'Required principal components: {PC_required}')

plt.figure()
plt.plot(range(1, len(rho)+1), rho, 'x-')
plt.plot(range(1, len(rho)+1), cumulative_rho, 'o-')
plt.plot([1, len(rho)], [threshold] * 2, 'b--')
plt.plot([PC_required] * 2, [0, 1], 'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative', 'Threshold', 'Required PC'])
plt.grid()
plt.show()

component = Vh.T[:, 0]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
          'lightgreen', 'pink', 'maroon',
          'teal', 'tan', 'fuchsia']
attribute_groups = names.copy()
attribute_groups.drop(columns=['education', 'result'], inplace=True)
color_map = dict(zip(attribute_groups, colors))

plt.figure()
for key, color in color_map.items():
    x_tics = [idx for idx, attr in enumerate(attributeNames) if key in attr]
    plt.bar(x_tics, component[x_tics], color=color)

plt.legend(attribute_groups, ncol=2, fontsize='small')
plt.show()
