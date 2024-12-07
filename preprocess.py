import pandas as pd
# import scipy
from sklearn.model_selection import train_test_split
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import seaborn as sns
# import matplotlib.pyplot as plt

fashion_df_train = pd.read_csv('data/fashion-mnist_train.csv')
fashion_df_test = pd.read_csv('data/fashion-mnist_test.csv')
fashion_df = fashion_df_train

# concatenate mnist together
for col in fashion_df:
  fashion_df[col].append(fashion_df_test[col])

# songs_df = pd.read_csv('data/songs.csv')

# generate synthetic data
synthetic_data = np.random.normal(100, 3, size=(10, 100))
synthetic_data_df = pd.DataFrame(synthetic_data)
synthetic_rows, synthetic_cols = synthetic_data_df.shape
# synthetic_data_df['labels'] = np.random.randint(1, 5, size=(synthetic_rows,1))

# make array of dfs
dfs = []

dfs.append(fashion_df)
# dfs.append(songs_df)
dfs.append(synthetic_data_df)

no_outliers_dfs = []
masked_dfs = []
imbalanced_dfs = []
removed_portions_masked = []


# generate batches with removed features, store removed
for df in dfs:
  for col in df:
    quartile_1 = np.percentile(df[col], 25, axis=0)
    quartile_3 = np.percentile(df[col], 75, axis=0)
    iqr = quartile_3 - quartile_1
    min_val = quartile_1 - 1.5*iqr
    max_val = quartile_3 + 1.5*iqr
        # iterate through array and trim outliers
    df = df[(df[col] >= min_val) & (df[col] <= max_val )]
  no_outliers_dfs.append(df)

for df in dfs:
  masked_dfs.append(df.mask(df > 4, 0))
  imbalanced_dfs.append(df.mask(df > 2, df/10**6))
  removed_portions_masked.append(df.where(df>2))

# partition into train/test
dfs_train = []
dfs_test = []

for df in dfs:
  train, test = train_test_split(df)
  dfs_train.append(train)
  dfs_test.append(test)

no_outliers_dfs_train = []
no_outliers_dfs_test = []

for df in no_outliers_dfs:
  train, test = train_test_split(df)
  no_outliers_dfs_train.append(train)
  no_outliers_dfs_test.append(test)

masked_dfs_train = []
masked_dfs_test = []

for df in masked_dfs:
  train, test = train_test_split(df)
  masked_dfs_train.append(train)
  masked_dfs_test.append(test)

imbalanced_dfs_train = []
imbalanced_dfs_test = []

for df in imbalanced_dfs:
  train, test = train_test_split(df)
  imbalanced_dfs_train.append(train)
  imbalanced_dfs_test.append(test)

# describe
for df in dfs_train:
  print(df.describe())

for df in dfs_test:
  print(df.describe())

for df in no_outliers_dfs_train:
  print(df.describe())

for df in no_outliers_dfs_test:
  print(df.describe())

for df in masked_dfs_train:
  print(df.describe())

for df in masked_dfs_test:
  print(df.describe())

for df in imbalanced_dfs_train:
  print(df.describe())

for df in imbalanced_dfs_test:
  print(df.describe())

# "link" to model








