#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy import stats
import random


# In[2]:


def loadYachtDataset():
    filename = "Datasets/yachtData.csv"
    return pd.read_csv(filename, header=None)


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
    pred = []
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
    return f_matrix, Y


# In[15]:


def plot_mean_rmse(rmse_train, rmse_test):
    plt.xlabel('power')
    plt.ylabel('Mean RMSE')
    plt.plot(rmse_train, 'r-', label='training set')
    plt.plot(rmse_test, 'b-', label='test set')
    plt.legend()
    plt.show()


# In[16]:


def ten_fold_split(dataset):
    folds = 10
    dataset_split = list()
    dataset_copy = list(range(len(dataset)))
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
        
    for i in range(folds):
        if(len(dataset_copy) != 0):
            dataset_split[i].append(dataset_copy.pop(0))      
        
    res = []
    for i in range(folds):
        train_indices = []
        test_indices = []
        for j in range(folds):
            if(i == j):
                test_indices.extend(dataset_split[j])
            else:
                train_indices.extend(dataset_split[j])
        res.append([train_indices, test_indices])
    return res


# In[17]:


def z_score_normalize_testset(x, means, stds):
    x = (x - means)/stds
    return x


# In[18]:


def z_score_normalize_trainset(x):
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    return (x - means) / stds, means, stds


# In[19]:


def main():
    df_data = loadYachtDataset()
    dataset = "Yacht"

    # root mean squared error
    RMSE_train = []
    RMSE_test = []
    SSE_train = []
    SSE_test = []
    
    print("Dataset: ", dataset, "\t\tSize of dataset: ", df_data.shape)
    for p in range(1, 8):
        print(p, '.')
        cnt = 1
        rmse_train_f = []
        rmse_test_f = []
        
        for train, test in ten_fold_split(df_data):
            org_train_data = np.take(df_data, train, 0).to_numpy()
            
            train_X, train_Y = construct_f_matrix(org_train_data, p)  
            rows_train, cols_train = train_X.shape
            
            # Initialize weight vector
            weights = np.zeros((cols_train + 1, 1))

            # z_score normalize train data
            scaled_train_X, scale_mean, scale_std = z_score_normalize_trainset(train_X)
            
            # Add column of constant feature
            train_X = np.concatenate((np.ones((rows_train, 1)), scaled_train_X), axis=1)
        
            # train
            weights, train_rmse, train_sse = fit_normal_equation(train_X, train_Y, weights)
            rmse_train_f.append(train_rmse)
            SSE_train.append(train_sse)

            org_test_data = np.take(df_data, test, 0).to_numpy()
            test_X, test_Y = construct_f_matrix(org_test_data, p)
            rows_test, cols_test = test_X.shape
            
            # z-score normalize test data using mean and standard deviation of training data
            scaled_test_X = z_score_normalize_testset(test_X, scale_mean, scale_std)
            

            # add constant feature to test data
            test_X = np.concatenate((np.ones((rows_test, 1)), scaled_test_X), axis=1)

            # test
            test_sse, test_rmse = predict(test_X, test_Y, weights)
            rmse_test_f.append(test_rmse)
            SSE_test.append(test_sse)

        mean_train_rmse = np.sum(rmse_train_f) / len(rmse_train_f)
        mean_test_rmse = np.sum(rmse_test_f) / len(rmse_test_f)
        RMSE_train.append(mean_train_rmse)
        RMSE_test.append(mean_test_rmse)

        print('\t\t    Mean RMSE')
        print('Train set\t', mean_train_rmse)
        print('Test set\t', mean_test_rmse)

    plot_mean_rmse(RMSE_train, RMSE_test)


# In[20]:


main()


# In[ ]:




