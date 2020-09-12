#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
from numpy import linalg


# In[2]:


def loadTwoSpiralDataset():
    filename ="Datasets/twoSpirals.csv"
    l_rate = 0.2
    n_iter = 500
    return pd.read_csv(filename, header=None), l_rate, n_iter


# In[3]:


def z_score_normalize_trainset(data):
    x, y = feature_result_split(data)
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    X = (x - means) / stds
    return np.concatenate((X, y), axis=1), means, stds


# In[4]:


def z_score_normalize_testset(data, means, stds):
    x, y = feature_result_split(data)
    x = (x - means)/stds
    return np.concatenate((x, y), axis=1)


# In[5]:


def feature_result_split(data):
    rows, cols = data.shape
    x = data[:, 0: cols-1]
    y = np.reshape((data[:, -1]), (rows, 1))
    return x, y


# In[6]:


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


# In[34]:


def train_weights_linear_kernel(train, l_rate, n_iter):
    x = train[:, 0: -1]
    y = train[:, -1]
    n_samples, n_features = x.shape
    alphas = [0 for i in range(n_samples)]
    bias = 0
    gram_matrix = linear(x, x.T)
    converge = False
    times = 0
    while not converge:
        converge = True
        for i in range(n_samples):
            error = np.sum(alphas * y * gram_matrix[i]) + bias
            if y[i] * error <= 0:
                converge = False
                alphas[i] += 1
                bias += y[i]
        times += 1
        if(times >= n_iter):
            break
    return alphas, bias


# In[8]:


def linear(x, z):
    return np.dot(x, z)


# In[9]:


def train_weights_RBF(train, l_rate, n_iter, sigma):
    x = train[:, 0: -1]
    y = train[:, -1]
    n_samples, n_features = x.shape
    alphas = [0 for i in range(n_samples)]
    bias = 0
    gram_matrix = np.zeros((n_samples, n_samples), dtype="float64")
    for i in range(n_samples):
        for j in range(n_samples):
            gram_matrix[i][j] = Gaussian(x[i], x[j], sigma)
    
    converge = False
    times = 0
    while not converge:
        converge = True
        for i in range(n_samples):
            pred = np.sum(alphas * y * gram_matrix[i]) + bias
            if y[i] * pred <= 0:
                converge = False
                alphas[i] += 1
                bias += y[i]
        times += 1
        if(times > n_iter):
            break
                
    weights = np.dot(alphas * y, x)
    weights = np.insert(weights, 0, bias, axis=0)
    return alphas, bias
    


# In[10]:


def Gaussian(x, z, sigma):
    v_sum = np.sum((x - z) ** 2)
    m = v_sum / (2 * (sigma ** 2))
    return np.exp(-m)


# In[11]:


def test_model_rbf(test, train, alphas, bias, sigma):
    x_train, y_train = feature_result_split(train)
    test_num, feature_num = test.shape
    y_pred = []
    for i in range(test_num):
        sum = 0
        for j in range(len(y_train)):
            sum += (alphas[j] * y_train[j] * Gaussian(x_train[j], test[i], sigma))
        if sum + bias >= 0:
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred


# In[12]:


def test_model(test, train, alphas, bias):
    x_train, y_train = feature_result_split(train)
    test_num, feature_num = test.shape
    y_pred = []
    for i in range(test_num):
        sum = 0;
        for j in range(len(y_train)):
            sum += (alphas[j] * y_train[j] * linear(x_train[j], test[i]))
        if sum + bias >= 0:
            y_pred.append(1)
        else:
            y_pred.append(-1)
#     for row in test:
#         prediction = predict(row, weights)
#         pred.append(prediction)
    return y_pred


# In[13]:


def predict(data, weights):
    activation = weights[0]
    for i in range(len(data) - 1):
        activation += weights[i + 1] * data[i]
    return 1 if activation >= 0 else -1


# In[39]:


def score_model(y_true, y_pred):
    k = len(np.unique(y_true))
    c_matrix = np.zeros((k, k))
    for i in range(len(y_true)):
        x_i = 0
        x_j = 0
        if y_true[i] == -1:
            x_i = 0
        else:
            x_i = 1
        if y_pred[i] == -1:
            x_j = 0
        else:
            x_j = 1
        c_matrix[x_i][x_j] += 1
    accuracy = (c_matrix[0][0] + c_matrix[1][1]) / len(y_true)
    if c_matrix[0][1] + c_matrix[1][1] == 0:
        precision = 0
    else:
        precision = c_matrix[1][1] / (c_matrix[0][1] + c_matrix[1][1])
    if c_matrix[1][0] + c_matrix[1][1] == 0:
        recall = 0
    else:
        recall = c_matrix[1][1] / (c_matrix[1][0] + c_matrix[1][1])
#     print('Confusion Matrix:')
#     print(c_matrix)
    return accuracy, precision, recall


# In[40]:


def linear_kernel():
    print('*' * 10 +'Linear Kernel' + '*' * 10 )
    df, l_rate, n_iter = loadTwoSpiralDataset()
    
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    
    for train, test in ten_fold_split(df):
        org_train_data = np.take(df, train, 0).to_numpy()
        org_test_data = np.take(df, test, 0).to_numpy()
        print('='*100)

        # normalize data
        train, scaled_mean, scaled_stds = z_score_normalize_trainset(org_train_data)

        # fit
        alphas, bias = train_weights_linear_kernel(train, l_rate, n_iter)
        
        # train data accuarcy, precision, recall
        print('Train set:')
        train_X, train_Y = feature_result_split(train)
        pred_train_y = test_model(train_X, train, alphas, bias)
        t_accuracy, t_precision, t_recall = score_model(train_Y, pred_train_y)
        train_accuracies.append(t_accuracy)
        train_precisions.append(t_precision)
        train_recalls.append(t_recall)
        print('Accuracy:', t_accuracy, '\tPrecision:', t_precision, '\tRecall:', t_recall)

        # normalize test data
        test = z_score_normalize_testset(org_test_data, scaled_mean, scaled_stds)
        test_X, test_Y = feature_result_split(test)

        # perceptron dual form
        print('\nTest set:')
        pred_y = test_model(test_X, train, alphas, bias)
        accuracy, precision, recall = score_model(test_Y, pred_y)
        test_accuracies.append(accuracy)
        test_precisions.append(precision)
        test_recalls.append(recall)
        print('Accuracy:', accuracy, '\tPrecision:', precision, '\tRecall:', recall)

    print('\n')
    print('#' * 80)
    print('Training Data:')
    print('Average accuracy: ', np.mean(train_accuracies), '\tStandard deviation:', np.std(train_accuracies))
    print('Average precision: ', np.mean(train_precisions), '\tStandard deviation:', np.std(train_precisions)) 
    print('Average recall: ', np.mean(train_recalls), '\tStandard deviation:', np.std(train_recalls))
    print('\nTest Data:')
    print('Average accuracy: ', np.mean(test_accuracies), '\tStandard deviation:', np.std(test_accuracies))
    print('Average precision: ', np.mean(test_precisions), '\tStandard deviation:', np.std(test_precisions)) 
    print('Average recall: ', np.mean(test_recalls), '\tStandard deviation:', np.std(test_recalls))
    print('#' * 80)


# In[41]:


def Gaussian_kernel():
    print('*' * 10 +'Gaussian Kernel' + '*' * 10 )
    df, l_rate, n_iter = loadTwoSpiralDataset()
    train_accuracies = [[] for y in range(6)]
    train_precisions = [[] for y in range(6)]
    train_recalls = [[] for y in range(6)]
    test_accuracies = [[] for y in range(6)]
    test_precisions = [[] for y in range(6)]
    test_recalls = [[] for y in range(6)]
    
    sigma_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    best_sigma = 0.01
    best_avg_accuracy = 0
    
    
    for train, test in ten_fold_split(df):
        org_train_data = np.take(df, train, 0).to_numpy()
        org_test_data = np.take(df, test, 0).to_numpy()
        for i in range(len(sigma_list)):
            sigma = sigma_list[i]
            # normalize data
            train, scaled_mean, scaled_stds = z_score_normalize_trainset(org_train_data)

            # fit
            alphas, bias = train_weights_RBF(train, l_rate, n_iter, sigma)
            
            # train data accuarcy, precision, recall
            train_X, train_Y = feature_result_split(train)
            pred_train_y = test_model_rbf(train_X, train, alphas, bias, sigma)
            t_accuracy, t_precision, t_recall = score_model(train_Y, pred_train_y)
            train_accuracies[i].append(t_accuracy)
            train_precisions[i].append(t_precision)
            train_recalls[i].append(t_recall)
            
            # normalize test data
            test = z_score_normalize_testset(org_test_data, scaled_mean, scaled_stds)
            test_X, test_Y = feature_result_split(test)
            pred_y = test_model_rbf(test_X, train, alphas, bias, sigma)
            accuracy, precision, recall = score_model(test_Y, pred_y)
            test_accuracies[i].append(accuracy)
            test_precisions[i].append(precision)
            test_recalls[i].append(recall)

    for i in range(6):
        print('#'*60)
        print('For sigma ', sigma_list[i])
        print('Average Train Accuracy:', train_accuracies[i])
        print('Average Train Precision:', train_precisions[i])
        print('Average Train Recall:', train_recalls[i])
        print('\nAverage Test Accuracy:', test_accuracies[i])
        print('Average Test Precision:', test_precisions[i])
        print('Average Test Recall:', test_recalls[i])
        print('\n')
    
    best_index = 0
    best_avg_test_ac = 0
    for i in range(len(test_accuracies)):
        if np.mean(test_accuracies[i]) > best_avg_test_ac:
            best_index = i
            best_avg_test_ac = np.mean(test_accuracies[i])
            
    print('Best sigma value:', sigma_list[best_index])
    print('\tAverage Train Accuracy:', np.mean(train_accuracies[best_index]), '\tStandard deviation:', np.std(train_accuracies[best_index]))
    print('\tAverage Train Precision:', np.mean(train_precisions[best_index]), '\tStandard deviation:', np.std(train_precisions[best_index]))
    print('\tAverage Train Recall:', np.mean(train_recalls[best_index]), '\tStandard deviation:', np.std(train_recalls[best_index]))
    print('\n\tAverage Test accuracy:', np.mean(test_accuracies[best_index]), '\tStandard deviation:', np.std(test_accuracies[best_index]))
    print('Average Test Precision:', np.mean(test_precisions[best_index]), '\tStandard deviation:', np.std(test_precisions[best_index]))
    print('Average Test Recall:', np.mean(test_recalls[best_index]),'\tStandard deviation:', np.std(test_recalls[best_index]))


# In[42]:


def main(model):
    if model == "l":
        linear_kernel()
    elif model == "g":
        Gaussian_kernel()
    else:
        print('Please enter correct kernel name.')
        return


# In[ ]:


import sys
if __name__ == '__main__':
    model = sys.argv[1]
    main(model)


# In[ ]:





# In[ ]:




