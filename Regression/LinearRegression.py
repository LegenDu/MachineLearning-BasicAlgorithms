#!/usr/bin/env python
# coding: utf-8

# In[85]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy import stats
import random
import sys
import statistics


# In[86]:


def loadHousingDataset():
    filename = "Datasets/housing.csv"
    lRate = 0.4 * (10 **(-3))
    tolerance = 0.5 * (10 ** (-2))
    return pd.read_csv(filename, header=None), lRate, tolerance

def loadYachtDataset():
    filename = "Datasets/yachtData.csv"
    lRate = 0.1 * (10 **(-2))
    tolerance = 0.1 * (10 ** (-2))
    return pd.read_csv(filename, header=None), lRate, tolerance

def loadConcreteDataset():
    filename = "Datasets/concreteData.csv"
    lRate = 0.7 * (10 **(-3))
    tolerance = 0.1 * (10 ** (-3))
    return pd.read_csv(filename, header=None), lRate, tolerance


# In[87]:


def feature_result_split(data):
    rows, cols = data.shape
    x = data[:, 0: cols-1]
    y = np.reshape((data[:, -1]), (rows, 1))
    return x, y


# In[88]:


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


# In[89]:


def z_score_normalize_testset(data, means, stds):
    x, y = feature_result_split(data)
    x = (x - means)/stds
    return np.concatenate((x, y), axis=1)


# In[90]:


def plot_training_rmses(rmses):
    plt.figure(1)
    plt.xlabel('Gradient Descent Iterations')
    plt.ylabel('RMSE (Training Set)')
    plt.plot(rmses)
    plt.show(block=True)


# In[91]:


def pred_y(x, weights):
    # Prediction initial value
    prediction = 0
    
    # Adding multiplication of each feature with it's weight
    for weight, feature in zip(weights, x):
        prediction += weight * feature
    
    return prediction


# In[92]:


def predict(data, weights):
    X, Y = feature_result_split(data)
    rows = data.shape[0]
    sses = []
    rmses = []
    for x, y in zip(X, Y):
        pred = pred_y(x, weights)
#         print('pred:', pred, ' y:', y)
        residuals = pred - y
        sse = np.sum((residuals ** 2))
        sses.append(sse)
    return np.sum(sses), math.sqrt(np.sum(sses) / rows)
#     return math.sqrt(np.sum(sses) / rows)


# In[93]:


def fit_gradient_descent(data, weights, max_iter_num, learningRate, tolerance, plot):
    rmses = []
    sses = []
    x, y = feature_result_split(data)
    rows = len(data)
    last_rmse = sys.maxsize
    
    for i in range(max_iter_num):
        y_pred = np.dot(x, weights)
        residuals = y_pred - y
        gradient_vector = np.dot(x.T, residuals)
        weights -= learningRate * gradient_vector
        sse = np.sum((residuals ** 2))
        sses.append(sse)
        rmse = math.sqrt(sse / rows)
        if len(rmses) != 0:
            last_rmse = rmses[-1]
        rmses.append(rmse)
        if(abs(last_rmse - rmse) < tolerance):
            break;
    if plot:
        plot_training_rmses(rmses)
    return weights, np.sum(rmses) / len(rmses), np.sum(sses) / len(sses)


# In[94]:


def fit_normal_equation(data, weights_ne):
    rows = data.shape[0]
    X, Y = feature_result_split(data)
    res_1 = np.linalg.inv(np.dot(X.T, X))
    res_2 = np.dot(res_1, X.T)
    weights_ne = np.dot(res_2, Y)
    y_pred = np.dot(X, weights_ne)
    residuals = y_pred - Y
    sse = np.sum((residuals ** 2))
    rmse = math.sqrt(sse / rows)
    return weights_ne, rmse, sse


# In[95]:


def z_score_normalize_trainset(data):
    x, y = feature_result_split(data)
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    X = (x - means) / stds
    return np.concatenate((X, y), axis=1), means, stds


# In[96]:


def plot_gd_vs_ne(gd_rmses, ne_rmses):
    plt.figure(2)
    plt.xlabel('Folds')
    plt.ylabel('RMSE')
    plt.plot(gd_rmses, 'r-', label='Gradient Descent')
    plt.plot(ne_rmses, 'b-', label='Normal Equation')
    plt.legend()
    plt.show()


# In[97]:


def main(dataset):
    # load dataset
    if dataset == 'Yacht':
        df_data, learningRate, tolerance = loadYachtDataset()
    elif dataset == 'Housing':
        df_data, learningRate, tolerance = loadHousingDataset()
    elif dataset == 'Concrete':
        df_data, learningRate, tolerance = loadConcreteDataset()
    else:
        print('Please input correct dataset name!')
        return
    
    # define maximum iterations
    max_iter_num = 1000

    cnt = 1

    # root mean squared error
    RMSE_train_gd = []
    RMSE_train_ne = []
    RMSE_test_gd = []
    RMSE_test_ne = []
    
    # sum of squared error
    SSE_train_gd = []
    SSE_train_ne = []
    SSE_test_gd = []
    SSE_test_ne = []

    # randomly chose a fold to plot gradient descent
    plot = random.randrange(1, 10)

    print("Dataset: ", dataset, "\t\tSize of dataset: ", df_data.shape, "\t\tPlot gradient descent of", plot, "fold")
    # for train, test in kfold.split(df_data):
    for train, test in ten_fold_split(df_data):
        print(cnt, ".")
        org_train_data = np.take(df_data, train, 0).to_numpy()

        # Initialize weight vector
        rows_train, cols = org_train_data.shape
        weights_gd = np.zeros((cols, 1))
        weights_ne = np.zeros((cols, 1))

        # z_score normalize train data
        scaled_train_data, scale_mean, scale_std = z_score_normalize_trainset(org_train_data)

        # Add constant feature to train data
        train_data = np.concatenate((np.ones((rows_train, 1)), scaled_train_data), axis=1)

        # train
        weights_gd, gd_train_rmse, gd_train_sse = fit_gradient_descent(train_data, weights_gd, max_iter_num, learningRate, tolerance, plot == cnt)
        weights_ne, ne_train_rmse, ne_train_sse = fit_normal_equation(train_data, weights_ne)
        RMSE_train_gd.append(gd_train_rmse)
        RMSE_train_ne.append(ne_train_rmse)
        SSE_train_gd.append(gd_train_sse)
        SSE_train_ne.append(ne_train_sse)

        org_test_data = np.take(df_data, test, 0).to_numpy()
        # z-score normalize test data using mean and standard deviation of training data
        rows_test = org_test_data.shape[0]
        scaled_test_data = z_score_normalize_testset(org_test_data, scale_mean, scale_std)

        # add constant feature to test data
        test_data = np.concatenate((np.ones((rows_test, 1)), scaled_test_data), axis=1)

        # test
        gd_test_sse, gd_test_rmse = predict(test_data, weights_gd)
        ne_test_sse, ne_test_rmse = predict(test_data, weights_ne)
#         gd_test_rmse = math.sqrt(np.sum(gd_test_sses) / len(gd_test_sses))
#         ne_test_rmse = math.sqrt(np.sum(ne_test_sses) / len(ne_test_sses))
        RMSE_test_gd.append(gd_test_rmse)
        RMSE_test_ne.append(ne_test_rmse)
        SSE_test_gd.append(gd_test_sse)
        SSE_test_ne.append(ne_test_sse)

        print('Gradient descent')
    #     print('Weights:',weights_gd.T)
        print('RMSE (training set): %.8f'% gd_train_rmse)
        print('RMSE (test set): %.8f'% gd_test_rmse)
        print('Normal Equation:')
    #     print('Weights:',weights_ne.T)
        print('RMSE (training set): %.8f'% ne_train_rmse)
        print('RMSE (test set): %.8f'% ne_test_rmse)
    #     print('='*80)
        print('\n')
        cnt+=1


    print('\nGradient descent')
    print('\t\t\t RMSE\t\t\t SSE \t\t\t Standard Deviation of SSE')
    print('Training set:\t', np.sum(RMSE_train_gd) / len(RMSE_train_gd),' \t', 
          np.sum(SSE_train_gd) / len(SSE_train_gd), '\t\t', statistics.stdev(SSE_train_gd))
    print('Test set:\t', np.sum(RMSE_test_gd) / len(RMSE_test_gd), '\t',
          np.sum(SSE_test_gd) / len(SSE_test_gd), '\t\t', statistics.stdev(SSE_test_gd))
    
    
#     print('Average train RMSE across 10 folds:', np.sum(RMSE_train_gd) / len(RMSE_train_gd))
#     print('Average test RMSE across 10 folds:', np.sum(RMSE_test_gd) / len(RMSE_test_gd))
#     print('Average train SSE across 10 folds:', np.sum(SSE_train_gd) / len(SSE_train_gd))
#     print('Standard deviation of train SSE across 10 folds:', statistics.stdev(SSE_train_gd))
#     print('Average test SSE across 10 folds:', np.sum(SSE_test_gd) / len(SSE_test_gd))
#     print('Standard deviation of test SSE across 10 folds:', statistics.stdev(SSE_test_gd))

    print('\nNormal equation')
    print('\t\t\t RMSE\t\t\t SSE \t\t\t Standard Deviation of SSE')
    print('Training set:\t', np.sum(RMSE_train_ne) / len(RMSE_train_ne),' \t', 
          np.sum(SSE_train_ne) / len(SSE_train_ne), '\t\t', statistics.stdev(SSE_train_ne))
    print('Test set:\t', np.sum(RMSE_test_ne) / len(RMSE_test_ne), '\t',
          np.sum(SSE_test_ne) / len(SSE_test_ne), '\t\t', statistics.stdev(SSE_test_ne))
    
    print('\n')
    
#     print('Average train RMSE across 10 folds:', np.sum(RMSE_train_ne) / len(RMSE_train_ne))
#     print('Average test RMSE across 10 folds:', np.sum(RMSE_test_ne) / len(RMSE_test_ne))
#     print('Standard deviation of train SSE across 10 folds:', statistics.stdev(SSE_train_ne))
#     print('Average train SSE across 10 folds:', np.sum(SSE_train_ne) / len(SSE_train_ne))
#     print('Average test SSE across 10 folds:', np.sum(SSE_test_ne) / len(SSE_test_ne))
#     print('Standard deviation of SSE across 10 folds:', statistics.stdev(SSE_test_ne))

    plot_gd_vs_ne(RMSE_train_gd, RMSE_train_ne)


# In[99]:


import sys
if __name__ == '__main__':
    dataset = sys.argv[1]
    main(dataset)




