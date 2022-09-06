import numpy as np
import pandas as pd


filename = 'adult.data'
df = pd.read_csv(filename, header=None)
names = pd.read_csv('header.data')

df.columns = names.columns

print(df[:5])

# (work class, marital status, occupation, relationship, race, native-country) change to 1 of K
# drop education( we have ed. number )
# (sex, result) to binary

