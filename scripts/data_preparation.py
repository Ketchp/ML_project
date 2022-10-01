import pandas as pd
import numpy as np
import os

# make sure this file can be imported from anywhere
orig_path = os.getcwd()
os.chdir(os.path.dirname(__file__))

filename = '../dataset/adult.data'
df = pd.read_csv(filename, sep=', ', header=None, engine='python')

names = pd.read_csv('../dataset/header.data')
df.columns = names.columns

# drop education( we have ed. number )
df.drop(columns='education', inplace=True)


# (sex, result) to binary
classNames = df['result'].unique()
C = len(classNames)
classDict = dict(zip(classNames, range(C)))

df['result'].replace(classDict, inplace=True)
df['sex'].replace({'Male': 0, 'Female': 1}, inplace=True)

df_nan = df.copy()

# drop missing values
df.replace({'?': np.nan}, inplace=True)
df.dropna(inplace=True)

df_wo_nan = df.copy()


def encode_one_of_k(dataframe: pd.DataFrame, column: str | list):
    if type(column) == str:
        from functools import reduce
        return reduce(encode_one_of_k, column, dataframe)

    one_hot = pd.get_dummies(dataframe[column])
    return dataframe.drop(columns=column).join(one_hot)


# aggregate small categories in attribute 'Native country'
df.rename(columns={'native-country': 'from-US'}, inplace=True)
country_cat = df['from-US'].unique()
country_mapping = dict(zip(country_cat,
                           np.zeros(len(country_cat), dtype=int)))
country_mapping['United-States'] = 1
df['from-US'].replace(country_mapping, inplace=True)


# drop final weight
df.drop(columns='fnlwgt', inplace=True)

df['relationship'].replace({'Husband': 'Married', 'Wife': 'Married'}, inplace=True)

# drop 99999 from capital gain: outliers
df = df[df['capital-gain'] < 50000]

# join capital gain and capital loss
df['capital-gain'] = df['capital-gain'] - df['capital-loss']
df.drop(columns='capital-loss', inplace=True)

df_nominal = df.copy()

# change to 1 of K: (work class, marital status, occupation, relationship, race)
nominal_columns = ['workclass', 'marital-status', 'occupation', 'relationship', 'race']
df = encode_one_of_k(df, nominal_columns)

attributeNames = df.columns.drop('result')

X = df[attributeNames].to_numpy()
y = df['result'].to_numpy()

N, M = X.shape

os.chdir(orig_path)
