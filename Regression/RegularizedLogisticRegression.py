#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy import stats
import random
import sys
import statistics


# In[2]:


def feature_result_split(data):
    rows, cols = data.shape
    x = data[:, 0: cols-1]
    y = np.reshape((data[:, -1]), (rows, 1))
    return x, y


# In[3]:


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


# In[4]:


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


# In[5]:


def fit_gradient_descent(X, Y, lRate, tolerance, n_iters, needPlot):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    bias = 0
    prev_cost = 0
    costs = []
    
    for i in range(n_iters):
        model = np.dot(X, weights) + bias
        y_pred = sigmoid(model)
        
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - Y))
        db = (1 / n_samples) * np.sum(y_pred - Y)
        
        weights -= lRate * dw
        bias -= lRate * db
        
#         # loss value
        cost = (-1 / n_samples) * (np.dot(Y.T, np.log(y_pred)) + (np.dot((1 - Y.T), np.log(1 - y_pred))))
        cost_value = cost[0][0]
        costs.append(cost_value)
        if abs(prev_cost - cost_value) < tolerance:
            break
        prev_cost = cost_value
    if needPlot:
        plot_gradient_descent_loss(costs, "Logistic Regression")
    return weights, bias


# In[6]:


def fit_regularized_gd(X, Y, lRate, tolerance, n_iters, needPlot, lambda_):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    bias = 0
    prev_cost = 0
    costs = []
    
    for i in range(n_iters):
        model = np.dot(X, weights) + bias
        y_pred = sigmoid(model)
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - Y)) + (lambda_ / n_samples) * weights
        db = (1 / n_samples) * np.sum(y_pred - Y)
        weights -= lRate * dw
        bias -= lRate * db
        cost = (-1 / n_samples) * (np.dot(Y.T, np.log(y_pred)) + (np.dot((1 - Y.T), np.log(1 - y_pred))))
        reg = np.sum((lambda_ / (2 * n_samples)) * (np.dot(weights, weights.T)))
        cost_value = cost[0][0] + reg
        costs.append(cost_value)
        if abs(prev_cost - cost_value) < tolerance:
            break
        prev_cost = cost_value
    if needPlot:
        plot_gradient_descent_loss(costs, "Regularized Logistic Regression")
    return weights, bias
    


# In[7]:


def plot_gradient_descent_loss(costs, name):
    plt.figure(2)
    plt.title(name)
    plt.ylabel('Logistic Loss')
    plt.plot(costs, 'r-')
    plt.show()


# In[8]:


def z_score_normalize_trainset(data):
    x, y = feature_result_split(data)
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    X = (x - means) / stds
    return np.concatenate((X, y), axis=1), means, stds


# In[9]:


def z_score_normalize_testset(data, means, stds):
    x, y = feature_result_split(data)
    x = (x - means)/stds
    return np.concatenate((x, y), axis=1)


# In[10]:


def predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
    return y_pred_class


# In[11]:


def plot_pred(y_true, y_pred, name):
    plt.figure(1)
    plt.title(name)
    plt.xlabel('instances')
    plt.ylabel('result')
    plt.plot(y_true, 'go', label='Actual Value')
    plt.plot(y_pred, 'ro', label='Predict Value')
    plt.legend()
    plt.show()


# In[12]:


def accuracy(y, pred_y):
    accuracy = np.sum(y == pred_y) / len(y)
    return accuracy


# In[13]:


def score_model(true, pred):
    true = true.astype(int)
    true = np.reshape(true, (1, len(true))).tolist()[0]
    
    k = len(np.unique(true))
    result = np.zeros((k, k))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    precision = result[1][1] / (result[1][1] + result[0][1])
    recall = result[1][1] / (result[1][1] + result[1][0])
    
    return result, precision, recall


# In[14]:


def loadSpambaseDataset():
    filename ="Datasets/spambase.csv"
    lRate = 0.2
    tolerance = 0.1 * (10 ** (-3))
    max_iters = 300
    lambda_ = 0.9
    return pd.read_csv(filename, header=None), lRate, tolerance, max_iters, lambda_

def loadDiabetesDataset():
    filename = "Datasets/diabetes.csv"
    lRate = 0.3
    tolerance = 0.1 * (10 ** (-4))
    max_iters = 300
    lambda_ = 0.9
    return pd.read_csv(filename, header=None), lRate, tolerance, max_iters, lambda_

def loadBreastCancerDataset():
    filename = "Datasets/breastcancer.csv"
    lRate = 0.3
    tolerance = 0.1 * (10 ** (-2))
    max_iters = 100
    lambda_ = 0.4
    return pd.read_csv(filename, header=None), lRate, tolerance, max_iters, lambda_


# In[15]:


def main(dataset):
    if dataset == "Spambase":
        df, lRate, tolerance, n_iters, lambda_ = loadSpambaseDataset()
    elif dataset == "Diabetes":
        df, lRate, tolerance, n_iters, lambda_ = loadDiabetesDataset()
    elif dataset == "BreastCancer":
        df, lRate, tolerance, n_iters, lambda_ = loadBreastCancerDataset()
    else:
        print('Please input correct dataset name.')
        return
    
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_confusion_matrix = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_confusion_matrix = []
    
    rg_train_accuracies = []
    rg_train_precisions = []
    rg_train_recalls = []
    rg_train_confusion_matrix = []
    rg_test_accuracies = []
    rg_test_precisions = []
    rg_test_recalls = []
    rg_test_confusion_matrix = []
    
    # randomly chose a fold to plot gradient descent
    plot = random.randrange(1, 10)
    
    cnt = 1
    for train, test in ten_fold_split(df):
        org_train_data = np.take(df, train, 0).to_numpy()
        org_test_data = np.take(df, test, 0).to_numpy()
        # fit model
        scaled_train_data, scaled_mean, scaled_stds = z_score_normalize_trainset(org_train_data)
        train_X, train_Y = feature_result_split(scaled_train_data)
        weights, bias = fit_gradient_descent(train_X, train_Y, lRate, tolerance, n_iters, plot == cnt)
        
        # get train data accuracy, precision and recall of normal logistic regression
        train_pred = predict(train_X, weights, bias)
        train_accuracies.append(accuracy(train_pred, train_Y.T))
        t_conf_matrix, t_precision, t_recall = score_model(train_Y, train_pred)
        train_confusion_matrix.append(t_conf_matrix)
        train_precisions.append(t_precision)
        train_recalls.append(t_recall)
        
        # scale test data
        scaled_test_data = z_score_normalize_testset(org_test_data, scaled_mean, scaled_stds)
        test_X, test_Y = feature_result_split(scaled_test_data)
        
        # get test data accuracy, precision and recall of normal logistic regression
        pred_test_y = predict(test_X, weights, bias)
        test_accuracies.append(accuracy(pred_test_y, test_Y.T))
        conf_matrix, precision, recall = score_model(test_Y, pred_test_y)
        test_confusion_matrix.append(conf_matrix)
        test_precisions.append(precision)
        test_recalls.append(recall)
        
        # train regularized logistic regression
        rg_weights, rg_bias = fit_regularized_gd(train_X, train_Y, lRate, tolerance, n_iters, plot == cnt, lambda_)
        
        # get train data accuracy, precision and recall of regularized logistic regression
        rg_train_pred = predict(train_X, rg_weights, rg_bias)
        rg_train_accuracies.append(accuracy(rg_train_pred, train_Y.T))
        rg_t_conf_matrix, rg_t_precision, rg_t_recall = score_model(train_Y, rg_train_pred)
        rg_train_confusion_matrix.append(rg_t_conf_matrix)
        rg_train_precisions.append(rg_t_precision)
        rg_train_recalls.append(rg_t_recall)
        
        # get test data accuracy, precision and recall of regularized logistic regression
        rg_pred_test_y = predict(test_X, rg_weights, rg_bias)
        rg_test_accuracies.append(accuracy(rg_pred_test_y, test_Y.T))
        rg_conf_matrix, rg_precision, rg_recall = score_model(test_Y, rg_pred_test_y)
        rg_test_confusion_matrix.append(rg_conf_matrix)
        rg_test_precisions.append(rg_precision)
        rg_test_recalls.append(rg_recall)
        
        if plot == cnt:
            plot_pred(test_Y, pred_test_y, "Logistic Regression")
            plot_pred(test_Y, rg_pred_test_y, "Regularized Logistic Regression")
        cnt += 1
#     conf_m = confusion_matrix[0]
#     for i in range(1, 10):
#         conf_m += confusion_matrix[i]
        
#     print('Confusion Matrix:\n', conf_m / 10)
    print('Normal Logistic Regression')
    print('Training Data:')
    print('Average accuracy: ', np.mean(train_accuracies), '\tStandard deviation:', np.std(train_accuracies))
    print('Average precision: ', np.mean(train_precisions), '\tStandard deviation:', np.std(train_precisions)) 
    print('Average recall: ', np.mean(train_recalls), '\tStandard deviation:', np.std(train_recalls))
    print('\nTest Data:')
    print('Average accuracy: ', np.mean(test_accuracies), '\tStandard deviation:', np.std(test_accuracies))
    print('Average precision: ', np.mean(test_precisions), '\tStandard deviation:', np.std(test_precisions)) 
    print('Average recall: ', np.mean(test_recalls), '\tStandard deviation:', np.std(test_recalls))
    
    print('\nRegularized Logistic Regression')
    print('Training Data:')
    print('Average accuracy: ', np.mean(rg_train_accuracies), '\tStandard deviation:', np.std(rg_train_accuracies))
    print('Average precision: ', np.mean(rg_train_precisions), '\tStandard deviation:', np.std(rg_train_precisions)) 
    print('Average recall: ', np.mean(rg_train_recalls), '\tStandard deviation:', np.std(rg_train_recalls))
    print('\nTest Data:')
    print('Average accuracy: ', np.mean(rg_test_accuracies), '\tStandard deviation:', np.std(rg_test_accuracies))
    print('Average precision: ', np.mean(rg_test_precisions), '\tStandard deviation:', np.std(rg_test_precisions)) 
    print('Average recall: ', np.mean(rg_test_recalls), '\tStandard deviation:', np.std(rg_test_recalls))


# In[35]:


import sys
if __name__ == '__main__':
    dataset = sys.argv[1]
    main(dataset)


# In[ ]:





# In[ ]:




