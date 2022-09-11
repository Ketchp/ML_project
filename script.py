import numpy as np
import pandas as pd


filename = 'adult.data'
df = pd.read_csv(filename, sep=', ',  header=None, engine='python')

names = pd.read_csv('header.data')
df.columns = names.columns

# drop education( we have ed. number )
df.drop(columns='education', inplace=True)

# (sex, result) to binary
df['result'].replace({'>50K': 0, '<=50K': 1}, inplace=True)
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

print(df)
