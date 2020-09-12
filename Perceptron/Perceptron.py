#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


# In[2]:


def loadPerceptronDataset():
    filename ="Datasets/perceptronData.csv"
    l_rate = 0.1
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


# In[7]:


def train_weights(train, l_rate, n_iter):
    weights = [0.0 for i in range(len(train[0]))]
    for i in range(n_iter):
#         sum_error = 0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
#             sum_error += error ** 2;
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    return weights


# In[8]:


def train_weights_dual_perceptron(train, l_rate, n_iter):
    x = train[:, 0: -1]
    y = train[:, -1]
    n_samples, n_features = x.shape
    alphas = [0 for i in range(n_samples)]
    bias = 0
    gram_matrix = np.dot(x, x.T)
    converge = False
    
#     for times in range(n_iter):
    while not converge:
        converge = True
        for i in range(n_samples):
            error = np.sum(alphas * y * gram_matrix[i]) + bias
            if y[i] * error <= 0:
                converge = False
                alphas[i] += 1
                bias += y[i]
                
    weights = np.dot(alphas * y, x)
    weights = np.insert(weights, 0, bias, axis=0)
    return weights


# In[9]:


def predict(data, weights):
    activation = weights[0]
    for i in range(len(data) - 1):
        activation += weights[i + 1] * data[i]
    return 1 if activation >= 0 else -1


# In[10]:


def test_model(test, weights):
    pred = []
    for row in test:
        prediction = predict(row, weights)
        pred.append(prediction)
    return pred


# In[11]:


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
    precision = c_matrix[1][1] / (c_matrix[0][1] + c_matrix[1][1])
    recall = c_matrix[1][1] / (c_matrix[1][0] + c_matrix[1][1])
#     print(c_matrix)
    return accuracy, precision, recall


# In[24]:


def main():
    df, l_rate, n_iter = loadPerceptronDataset()
    i = 0
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    
    train_accuracies_dual = []
    train_precisions_dual = []
    train_recalls_dual = []
    
    test_accuracies_dual = []
    test_precisions_dual = []
    test_recalls_dual = []
    
    for train, test in ten_fold_split(df):
        print('='*80)
        print(i+1, "Fold.")
        org_train_data = np.take(df, train, 0).to_numpy()
        org_test_data = np.take(df, test, 0).to_numpy()

        # normalize train data
        train, scaled_mean, scaled_stds = z_score_normalize_trainset(org_train_data)

        # fit
        weights = train_weights(train, l_rate, n_iter)
        weights_dual = train_weights_dual_perceptron(train, l_rate, n_iter)

        # normalize test data
        test = z_score_normalize_testset(org_test_data, scaled_mean, scaled_stds)
        test_Y = test[:, -1]
        train_Y = train[:, -1]

        # normal perceptron
        print('Perceptron:')
        pred_train_y = test_model(train, weights)
        t_accuracy, t_precision, t_recall = score_model(train_Y, pred_train_y)
        train_accuracies.append(t_accuracy)
        train_precisions.append(t_precision)
        train_recalls.append(t_recall)
        print('Train set:\tAccuracy:', t_accuracy, '\tPrecision:', t_precision, '\tRecall:', t_recall)
        
        pred_y = test_model(test, weights)
        accuracy, precision, recall = score_model(test_Y, pred_y)
        test_accuracies.append(accuracy)
        test_precisions.append(precision)
        test_recalls.append(recall)
        print('Test set:\tAccuracy:', accuracy, '\tPrecision:', precision, '\tRecall:', recall)

        # perceptron dual form
        print('Perceptron (dual form):')
        pred_train_y_dual = test_model(train, weights_dual)
        t_accuracy_dual, t_precision_dual, t_recall_dual = score_model(train_Y, pred_train_y_dual)
        train_accuracies_dual.append(t_accuracy_dual)
        train_precisions_dual.append(t_precision_dual)
        train_recalls_dual.append(t_recall_dual)
        print('Train set:\tAccuracy:', t_accuracy_dual, '\tPrecision:', t_precision_dual, '\tRecall:', t_recall_dual)
        
        pred_y_dual = test_model(test, weights_dual)
        accuracy_dual, precision_dual, recall_dual = score_model(test_Y, pred_y_dual)
        test_accuracies_dual.append(accuracy_dual)
        test_precisions_dual.append(precision_dual)
        test_recalls_dual.append(recall_dual)
        print('Test set:\tAccuracy:', accuracy_dual, '\tPrecision:', precision_dual, '\tRecall:', recall_dual)
        
        i += 1
    print('*'*80)
    print('Perceptron')
    print('Training Data:')
    print('Average accuracy: ', np.mean(train_accuracies), '\tStandard deviation:', np.std(train_accuracies))
    print('Average precision: ', np.mean(train_precisions), '\tStandard deviation:', np.std(train_precisions)) 
    print('Average recall: ', np.mean(train_recalls), '\tStandard deviation:', np.std(train_recalls))
    print('\nTest Data:')
    print('Average accuracy: ', np.mean(test_accuracies), '\tStandard deviation:', np.std(test_accuracies))
    print('Average precision: ', np.mean(test_precisions), '\tStandard deviation:', np.std(test_precisions)) 
    print('Average recall: ', np.mean(test_recalls), '\tStandard deviation:', np.std(test_recalls))
    print('\n')
    print('Dual Perceptron')
    print('Training Data:')
    print('Average accuracy: ', np.mean(train_accuracies_dual), '\tStandard deviation:', np.std(train_accuracies_dual))
    print('Average precision: ', np.mean(train_precisions_dual), '\tStandard deviation:', np.std(train_precisions_dual)) 
    print('Average recall: ', np.mean(train_recalls_dual), '\tStandard deviation:', np.std(train_recalls_dual))
    print('\nTest Data:')
    print('Average accuracy: ', np.mean(test_accuracies_dual), '\tStandard deviation:', np.std(test_accuracies_dual))
    print('Average precision: ', np.mean(test_precisions_dual), '\tStandard deviation:', np.std(test_precisions_dual)) 
    print('Average recall: ', np.mean(test_recalls_dual), '\tStandard deviation:', np.std(test_recalls_dual))


# In[25]:


if __name__ == '__main__':
    main()


# In[ ]:




