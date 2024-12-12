import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import synthetic_data_df_train, synthetic_data_df_test 
from sklearn.datasets import make_blobs

pd.set_option("display.max.columns", None)

#Uncomment to look at synthetic data
#print(synthetic_data_df_train)
#print(synthetic_data_df_test)


#full synthetic dataset

Xtrain_full = synthetic_data_df_train.iloc[:, :-1]
ytrain_full = synthetic_data_df_train.iloc[:, -1:]

Xtest_full = synthetic_data_df_test.iloc[:, :-1]
ytest_full = synthetic_data_df_test.iloc[:, -1:]


#Uncomment to look at full data set

#print("Xtrain matrix:")
#print(Xtrain_full)
#rows, cols = Xtrain_full.shape
#print(rows, cols)

#print("Xtest matrix:")
#print(Xtest_full)
#rows, cols = Xtest_full.shape
#print(rows, cols)

#print("ytrain matrix:")
#print(ytrain_full)
#rows, cols = ytrain_full.shape
#print(rows, cols)

#print("ytest matrix:")
#print(ytest_full)
#rows, cols = ytest_full.shape
#print(rows, cols)


#randomly remove values from synthetic dataset

np.random.seed(50)

num_missing_train = int(synthetic_data_df_train.size * 0.2)
num_missing_test = int(synthetic_data_df_test.size * 0.2)


random_indices_train = np.random.choice(synthetic_data_df_train.size, num_missing_train, replace=False)
random_indices_test = np.random.choice(synthetic_data_df_test.size, num_missing_test, replace=False)
flat_train = synthetic_data_df_train.values.flatten()
flat_test = synthetic_data_df_test.values.flatten()
flat_train[random_indices_train] = np.nan
flat_test[random_indices_test] = np.nan

synthetic_df_train_missing = pd.DataFrame(flat_train.reshape(synthetic_data_df_train.shape), columns=synthetic_data_df_train.columns)
synthetic_df_test_missing = pd.DataFrame(flat_test.reshape(synthetic_data_df_test.shape), columns=synthetic_data_df_test.columns)

Xtrain_missing = synthetic_df_train_missing.iloc[:, :-1]
ytrain_missing = synthetic_df_train_missing.iloc[:, -1:]

Xtest_missing = synthetic_df_test_missing.iloc[:, :-1]
ytest_missing = synthetic_df_test_missing.iloc[:, -1:]



#impute using mean
Xtrain_means = Xtrain_missing.mean()
Xtrain_means = Xtrain_means.fillna(0)

Xtest_means = Xtest_missing.mean()
Xtest_means = Xtest_means.fillna(0)

ytrain_means = ytrain_missing.mean()
ytrain_means = ytrain_means.fillna(0)

ytest_means = ytest_missing.mean()
ytest_means = ytest_means.fillna(0)



#impute using means --- Change to mode or deletion to see affects of mode or deletion imputatiom

Xtrain_missing= Xtrain_missing.fillna(Xtrain_means)
ytrain_missing = ytrain_missing.fillna(ytrain_means)
Xtest_missing = Xtest_missing.fillna(Xtest_means)
ytest_missing = ytest_missing.fillna(ytest_means)


#Uncomment to look at missing dataset with imputed values
#print("Xtrain missing matrix:")
#print(Xtrain_missing)
#rows, cols = Xtrain_missing.shape
#print(rows, cols)

#print("Xtest missing matrix:")
#print(Xtest_missing)
#rows, cols = Xtest_missing.shape
#print(rows, cols)

#print("ytrain missing matrix:")
#print(ytrain_missing)
#rows, cols = ytrain_missing.shape
#print(rows, cols)

#print("ytest missing matrix:")
#print(ytest_missing)
#rows, cols = ytest_missing.shape
#print(rows, cols)

Xtrain_missing = Xtrain_missing.to_numpy()
Xtest_missing = Xtest_missing.to_numpy()
ytrain_missing = ytrain_missing.to_numpy()
ytest_missing = ytest_missing.to_numpy()

Xtrain_full = Xtrain_full.to_numpy()
Xtest_full = Xtest_full.to_numpy()
ytrain_full = ytrain_full.to_numpy()
ytest_full = ytest_full.to_numpy()


# Normalizing full
mean_vec = np.mean(Xtrain_full, axis=0)
std_vec = np.std(Xtrain_full, axis=0)

Xtrain_normalized = (Xtrain_full - mean_vec) / std_vec
Xtest_normalized = (Xtest_full - mean_vec) / std_vec

ytrain_mean = np.mean(ytrain_full)
ytrain_std = np.std(ytrain_full)

ytrain_normalized = (ytrain_full - ytrain_mean) / ytrain_std
ytest_normalized = (ytest_full - ytrain_mean) / ytrain_std

xx = np.arange(-5, 21, 1) 
lambda_vec = np.exp(xx) 

# Train ridge regression model for various lambda  values for full dataset
coefficient_mat = np.zeros((Xtrain_normalized.shape[1], len(lambda_vec)))

for i in range(len(lambda_vec)):
    I = np.eye(Xtrain_normalized.shape[1]) 
    w = np.linalg.inv((Xtrain_normalized.T @ Xtrain_normalized) + (lambda_vec[i] * I)) @ (Xtrain_normalized.T @ ytrain_normalized)
    #print(w)
    coefficient_mat[:, i] = np.array(w).flatten()


#Plotting different lambdas
plt.figure()
for i in range(coefficient_mat.shape[0]):
    plt.plot(xx, coefficient_mat[i, :], label=['Feature ' + str(i)])

plt.xlabel('ln(lambda)')
plt.ylabel('Feature coefficient values')
plt.title('Feature coefficient values for various regularization amounts for full')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# MSE Plotting and predicting

mse_test_vec = []
mse_train_vec = []

for i in range(len(lambda_vec)):
    w = coefficient_mat[:, i] 
    
    ytest_pred = Xtest_normalized @ w 
    msetest = np.mean((ytest_normalized - ytest_pred) ** 2) 

    ytrain_pred = Xtrain_normalized @ w
    msetrain = np.mean((ytrain_normalized - ytrain_pred) ** 2) 

    mse_test_vec.append(msetest)
    mse_train_vec.append(msetrain)

min_msetest_full = min(mse_test_vec)
plt.figure()
plt.plot(xx, mse_test_vec, '-o', label='Test MSE')
plt.plot(xx, mse_train_vec, '-x', label='Train MSE')
plt.xlabel('ln(lambda)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE as a function of ln(lambda) for full')
plt.legend(loc='upper right')
plt.grid()
plt.show()



#REPEAT FOR MISSING


# Normalizing
mean_vec = np.mean(Xtrain_missing, axis=0)
std_vec = np.std(Xtrain_missing, axis=0)

print("Xtrain mean_vec")
print(mean_vec)

Xtrain_normalized = (Xtrain_missing - mean_vec) / std_vec
Xtest_normalized = (Xtest_missing - mean_vec) / std_vec

ytrain_mean = np.mean(ytrain_missing)
ytrain_std = np.std(ytrain_missing)

ytrain_normalized = (ytrain_missing - ytrain_mean) / ytrain_std
ytest_normalized = (ytest_missing - ytrain_mean) / ytrain_std

xx = np.arange(-5, 21, 1) 
lambda_vec = np.exp(xx) 



# Train ridge regression model for various lambda (regularization parameter) values
coefficient_mat = np.zeros((Xtrain_normalized.shape[1], len(lambda_vec)))

for i in range(len(lambda_vec)):
    I = np.eye(Xtrain_normalized.shape[1]) 
    w = np.linalg.inv((Xtrain_normalized.T @ Xtrain_normalized) + (lambda_vec[i] * I)) @ (Xtrain_normalized.T @ ytrain_normalized)
    #print(w)
    coefficient_mat[:, i] = np.array(w).flatten()



#Plotting different lambdas
plt.figure()
for i in range(coefficient_mat.shape[0]):
    plt.plot(xx, coefficient_mat[i, :], label=['Feature ' + str(i)])


plt.xlabel('ln(lambda)')
plt.ylabel('Feature coefficient values')
plt.title('Feature coefficient values for various regularization amounts for missing')
plt.legend(loc='upper right')
plt.grid()
plt.show()



# MSE Plotting and predicting on Test
#w = coefficient_mat[:, 0] 
#y_pred = Xtest_normalized @ w 
#mse = np.mean((ytest_normalized - y_pred) ** 2) 

mse_test_vec = []
mse_train_vec = []

for i in range(len(lambda_vec)):
    w = coefficient_mat[:, i] 
    
    ytest_pred = Xtest_normalized @ w 
    msetest = np.mean((ytest_normalized - ytest_pred) ** 2) 

    ytrain_pred = Xtrain_normalized @ w
    msetrain = np.mean((ytrain_normalized - ytrain_pred) ** 2) 

    mse_test_vec.append(msetest)
    mse_train_vec.append(msetrain)

min_msetest_missing = min(mse_test_vec)

plt.figure()
plt.plot(xx, mse_test_vec, '-o', label='Test MSE')
plt.plot(xx, mse_train_vec, '-x', label='Train MSE')
plt.xlabel('ln(lambda)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE as a function of ln(lambda) for missing')
plt.legend(loc='upper right')
plt.grid()
plt.show()


print("Min MSE Test for full dataset: ")
print(min_msetest_full)
print("Min MSE Test for Missing Value dataset: ")
print(min_msetest_missing)
percent_error = ((abs(min_msetest_missing - min_msetest_full)) / min_msetest_full) * 100
print('Percent error with respect to full dataset')
print(percent_error)




