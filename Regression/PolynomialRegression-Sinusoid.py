#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy import stats


# In[2]:


def loadSinusoidDataset():
    trainData_filename = "Datasets/sinData_Train.csv"
    testData_filename = "Datasets/sinData_Validation.csv"
    return pd.read_csv(trainData_filename, header=None), pd.read_csv(testData_filename, header=None)


# In[3]:


def feature_result_split(data):
    rows, cols = data.shape
    x = data[:, 0: cols-1]
    y = np.reshape((data[:, -1]), (rows, 1))
    return x, y


# In[4]:


def pred_y(x, weights):
    # Prediction initial value
    prediction = 0
    
    # Adding multiplication of each feature with it's weight
    for weight, feature in zip(weights, x):
        prediction += weight * feature
    
    return prediction


# In[5]:


def fit_normal_equation(X, Y, weights):
    rows = X.shape[0]
#     X, Y = construct_f_matrix(data, p)
    res_1 = np.linalg.inv(np.dot(X.T, X))
    res_2 = np.dot(res_1, X.T)
    weights = np.dot(res_2, Y)
    y_pred = np.dot(X, weights)
    residuals = y_pred - Y
    sse = np.sum((residuals ** 2))
    rmse = math.sqrt(sse / rows)
    return weights, rmse, sse


# In[6]:


def predict(X, Y, weights):
    rows = X.shape[0]
    sses = []
    rmses = []
    for x, y in zip(X, Y):
        pred = pred_y(x, weights)
#         print('pred:', pred, ' y:', y)
        residuals = pred - y
        sse = np.sum((residuals ** 2))
        sses.append(sse)
    return np.sum(sses), math.sqrt(np.sum(sses) / rows)


# In[7]:


def construct_f_matrix(data, p):
    X, Y = feature_result_split(data)
    f_matrix = X
    for i in range(1, p):
        f_matrix = np.concatenate((f_matrix, np.power(X, i+1)), axis=1)
    f_matrix = np.concatenate((np.ones((len(f_matrix), 1)), f_matrix), axis=1)
    return f_matrix, Y


# In[8]:


def plot_sse(train_sse, test_sse):
    plt.xlabel('power')
    plt.ylabel('SSE')
    plt.plot(train_sse, 'r-', label='training set')
    plt.plot(test_sse, 'b-', label='test set')
    plt.legend()
    plt.show()


# In[9]:


def main():
    org_train_data, org_test_data = loadSinusoidDataset()
    dataset = "Sinusoid"

    # root mean squared error
    RMSE_train = []
    RMSE_test = []
    SSE_train = []
    SSE_test = []

    print("Dataset: ", dataset, "\t\tSize of training dataset: ", org_train_data.shape, "\t\tSize of validation dataset: ", org_test_data.shape)

    train_data = org_train_data.to_numpy()
    test_data = org_test_data.to_numpy()
    
    for i in range(1, 16):
        print(i, '.')
        rows_train, cols = train_data.shape
        # Initialize weight vector
        weights = np.zeros((i + 1, 1))

        # train
        train_X, train_Y = construct_f_matrix(train_data, i)
        weights, train_rmse, train_sse = fit_normal_equation(train_X, train_Y, weights)
        mean_train_sse = train_sse / rows_train
        RMSE_train.append(train_rmse)
        SSE_train.append(mean_train_sse)

        rows_test = org_test_data.shape[0]

        # test
        test_X, test_Y = construct_f_matrix(test_data, i)
        test_sse, test_rmse = predict(test_X, test_Y, weights)
        mean_test_sse = test_sse / rows_test
        RMSE_test.append(test_rmse)
        SSE_test.append(mean_test_sse)
        
        print('\t\t  RMSE\t\t\t\t  Mean SSE')
        print('Train set\t', train_rmse, '\t\t', mean_train_sse)
        print('Test set\t', test_rmse, '\t\t', mean_test_sse)
        

#         print('Weights:',weights.T)
#         print('RMSE (training set): ', train_rmse)
#         print('RMSE (test set):', test_rmse)
#         print('Mean SSE (training set):', mean_train_sse)
#         print('Mean SSE (test set):', mean_test_sse)
        print('='*80)
    
    plot_sse(SSE_train, SSE_test)
    print('Average train rmse:', np.sum(RMSE_train) / len(RMSE_train))
    print('Average validation set rmse:', np.sum(RMSE_test) / len(RMSE_test))
    print('Average train sse:', np.sum(SSE_train) / len(SSE_train))
    print('Average validation set sse:', np.sum(SSE_test) / len(SSE_test))

    


# In[10]:


main()


# In[ ]:





# In[ ]:




