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


def loadSinusoidDataset():
    filename = "Datasets/sinData_Train.csv"
    return pd.read_csv(filename, header=None)


# In[3]:


def feature_result_split(data):
    rows, cols = data.shape
    x = data[:, 0: cols-1]
    y = np.reshape((data[:, -1]), (rows, 1))
    return x, y


# In[4]:


def fit(data, weights, w0, l):
    rows = data.shape[0]
    X, Y = feature_result_split(data)
    # calculate the weights
    X_square = np.dot(X.T, X)
    lambda_I = np.identity(X.shape[1]) * l
    part_1 = np.linalg.inv(X_square + lambda_I)
    part_2 = np.dot(part_1, X.T)
    weights = np.dot(part_2, Y)
    
    # calculate RMSE of training data
    y_pred = np.dot(X, weights)
    y_pred = y_pred + w0
    residuals = y_pred - Y
    sse = np.sum((residuals ** 2))
    rmse = math.sqrt(sse / rows)
    return weights, rmse


# In[5]:


def pre_process_data(data, p):
    X, Y = construct_f_matrix(data, p)
    p_data = np.concatenate((X, Y), axis=1)
    col_means = np.mean(p_data, axis=0)
    p_data = p_data - col_means
    return p_data, col_means


# In[6]:


def construct_f_matrix(data, p):
    X, Y = feature_result_split(data)
    f_matrix = X
    for i in range(1, p):
        f_matrix = np.concatenate((f_matrix, np.power(X, i+1)), axis=1)
    return f_matrix, Y


# In[7]:


def predict(data, weights, w0):
    X, Y = feature_result_split(data)
    rows = X.shape[0]
    sses = []
    for x, y in zip(X, Y):
        pred = pred_y(x, weights, w0)
        residuals = pred - y
        sse = np.sum((residuals ** 2))
        sses.append(sse)
    return math.sqrt(np.sum(sses) / rows)


# In[8]:


def pred_y(x, weights, w0):
    # Prediction initial value
    prediction = 0
    
    # Adding multiplication of each feature with it's weight
    for weight, feature in zip(weights, x):
        prediction += weight * feature
    
    return prediction + w0


# In[9]:


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


# In[18]:


def plot_rmse_vs_lambda(no, rmse, color, p, plabel):
    plt.figure(no)
    plt.xlabel('lambda (p=' + str(p) + ')')
    plt.ylabel('Average RMSE')
    plt.xlim([0, 10])
    plt.plot(rmse, color, label=plabel)
    plt.legend()
    plt.show()


# In[19]:


def main(power):
    df = loadSinusoidDataset()
    dataset = "Sinusoid"
    
    # root mean squared error
    avg_RMSE_train = []
    avg_RMSE_test = []

    for i in range(0, 101, 2):
        lambda_ = i / 10
        RMSE_Train = []
        RMSE_Test = []
        
        for train, test in ten_fold_split(df):
#             print('*'*100)
            org_train_data = np.take(df, train, 0).to_numpy()
            org_test_data = np.take(df, test, 0).to_numpy()

            rows_train, cols_train = org_train_data.shape

            # center the data
            train_data, train_col_means = pre_process_data(org_train_data, power)
            test_data, test_col_means = pre_process_data(org_test_data, power)

            # initialize theta
            weights = np.zeros((cols_train, 1))
            # set w0 to the mean value of target variable y
            w0 = train_col_means[-1]
            
            # fit
            weights, rmse_train = fit(train_data, weights, w0, lambda_)
            RMSE_Train.append(rmse_train)
            
            rmse_test = predict(test_data, weights, w0)
            RMSE_Test.append(rmse_test)
                        
        avg_RMSE_train.append(np.sum(RMSE_Train) / len(RMSE_Train))
        avg_RMSE_test.append(np.sum(RMSE_Test)/ len(RMSE_Test))
    print('avg train rmse:', avg_RMSE_train)
    print('avg test rmse:', avg_RMSE_test)
    plot_rmse_vs_lambda(0, avg_RMSE_train, 'b-', power, 'train')
    plot_rmse_vs_lambda(1, avg_RMSE_test, 'r-', power, 'test')
    
        


# In[16]:


import sys
if __name__ == '__main__':
    p = sys.argv[1]
    main(int(p))


# In[ ]:




