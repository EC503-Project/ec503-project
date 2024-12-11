import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import titanic_df_train, titanic_df_test
from sklearn.datasets import make_blobs

pd.set_option("display.max.columns", None)

# more data processing
titanic_df_train['Sex'] = titanic_df_train['Sex'].map({'female': 1, 'male': 0})
titanic_df_train['Embarked'] = titanic_df_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

titanic_df_test['Sex'] = titanic_df_test['Sex'].map({'female': 1, 'male': 0})
titanic_df_test['Embarked'] = titanic_df_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})


#print(titanic_df_train.head)

feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
y_feature = ['Survived_x']

Xtrain = titanic_df_train[feature_columns]
ytrain = titanic_df_train[y_feature]

Xtest = titanic_df_test[feature_columns]
ytest = titanic_df_test[y_feature]



#imputation using mean, change to mode or deletion to see affects of different imputation

for column in Xtrain.columns:
    Xtrain[column] = Xtrain[column].fillna(Xtrain[column].mean())

for column in ytrain.columns:
    ytrain[column] = ytrain[column].fillna(ytrain[column].mean())

for column in Xtest.columns:
    Xtest[column] = Xtest[column].fillna(Xtest[column].mean())

for column in ytest.columns:
    ytest[column] = ytest[column].fillna(ytest[column].mean())

Xtrain = Xtrain.to_numpy()
Xtest = Xtest.to_numpy()
ytrain = ytrain.to_numpy()
ytest = ytest.to_numpy()


#Uncomment to look at datasets
#print("Xtrain matrix:")
#print(Xtrain)
#rows, cols = Xtrain.shape
#print(rows, cols)

#print("ytrain matrix:")
#print(ytrain)
#rows, cols = ytrain.shape
#print(rows, cols)

#print("ytest matrix:")
#print(ytest)
#rows, cols = ytest.shape
#print(rows, cols)

#print("ytest matrix:")
#print(ytest)
#rows, cols = ytest.shape
#print(rows, cols)


# Normalizing
mean_vec = np.mean(Xtrain, axis=0)
std_vec = np.std(Xtrain, axis=0)

Xtrain_normalized = (Xtrain - mean_vec) / std_vec
Xtest_normalized = (Xtest - mean_vec) / std_vec

ytrain_mean = np.mean(ytrain)
ytrain_std = np.std(ytrain)

ytrain_normalized = (ytrain - ytrain_mean) / ytrain_std
ytest_normalized = (ytest - ytrain_mean) / ytrain_std


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
plt.title('Feature coefficient values for various regularization amounts')
plt.legend(loc='upper right')
plt.grid()
plt.show()


# MSE Plotting and predicting on Test
mse_test_vec = []
mse_train_vec = []

for i, lam in enumerate(lambda_vec):
    w = coefficient_mat[:, i] 

    ytest_pred = Xtest_normalized @ w 
    msetest = np.mean((ytest_normalized - ytest_pred) ** 2) 

    ytrain_pred = Xtrain_normalized @ w 
    msetrain = np.mean((ytrain_normalized - ytrain_pred) ** 2) 

    mse_test_vec.append(msetest)
    mse_train_vec.append(msetrain)


plt.figure()
plt.plot(xx, mse_test_vec, '-o', label = 'Test MSE')
plt.plot(xx, mse_train_vec, '-x', label='Train MSE')
plt.xlabel('ln(lambda)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE as a function of ln(lambda)')
plt.legend(loc='upper right')
plt.grid()
plt.show()

