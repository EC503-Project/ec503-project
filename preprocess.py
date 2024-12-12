import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data_scaler = MinMaxScaler()

fashion_df_train = pd.read_csv('fashion-mnist_train.csv')
fashion_df_test = pd.read_csv('fashion-mnist_test.csv')
fashion_df = pd.concat([fashion_df_train, fashion_df_test])
fashion_df.head()
fashion_df.describe()
data_scaler.fit(fashion_df)
fashion_df = data_scaler.transform(fashion_df)
fashion_df = pd.DataFrame(fashion_df)
fashion_df_train, fashion_df_test = train_test_split(fashion_df)

# generate synthetic data
synthetic_data = np.random.normal(100, 3, size=(10, 100))
synthetic_data_df = pd.DataFrame(synthetic_data)
synthetic_rows, synthetic_cols = synthetic_data_df.shape
synthetic_data_df.head()
synthetic_data_df.describe()
data_scaler.fit(synthetic_data_df)
synthetic_data_df = data_scaler.transform(synthetic_data_df)
synthetic_data_df[0] = np.random.randint(1, 5, size=(1, synthetic_cols))
synthetic_data_df = pd.DataFrame(synthetic_data_df)
synthetic_data_df_train, synthetic_data_df_test = train_test_split(synthetic_data_df)
# make array of dfs
dfs = []

dfs.append(fashion_df)
dfs.append(synthetic_data_df)

no_outliers_dfs = []
masked_dfs = []
outlier_dfs = []
removed_portions_masked = []

# generate batches with removed features, store removed
for df in dfs:
  for col in df.columns:
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
  outlier_dfs.append(df.mask(df > 2, df/10**4))
  removed_portions_masked.append(df.where(df>2))

for df in masked_dfs:
    print(df.head())
    print(df.describe())

for df in outlier_dfs:
    print(df.head())
    print(df.describe())

for df in removed_portions_masked:
    print(df.head())
    print(df.describe())


titanic_df = pd.read_csv('titanic.csv')
titanic_df_train, titanic_df_test = train_test_split(titanic_df)

books_df = pd.read_csv('Books.csv')
ratings_df = pd.read_csv('Ratings.csv')
books_df = pd.merge(books_df, ratings_df, how='left', on='ISBN')
books_df_train, books_df_test = train_test_split(books_df)