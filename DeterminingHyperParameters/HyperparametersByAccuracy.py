#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import random
import sys
from sklearn.svm import SVC


# In[2]:


def loadSpambaseDataset():
    filename ="Datasets/spambase.csv"
    lRate = 0.2
    tolerance = 0.1 * (10 ** (-3))
    max_iters = 300
    return pd.read_csv(filename, header=None), lRate, tolerance, max_iters

def loadDiabetesDataset():
    filename = "Datasets/diabetes.csv"
    lRate = 0.3
    tolerance = 0.1 * (10 ** (-4))
    max_iters = 300
    return pd.read_csv(filename, header=None), lRate, tolerance, max_iters

def loadBreastCancerDataset():
    filename = "Datasets/breastcancer.csv"
    lRate = 0.3
    tolerance = 0.1 * (10 ** (-2))
    max_iters = 100
    return pd.read_csv(filename, header=None), lRate, tolerance, max_iters


# In[3]:


def feature_result_split(data):
    rows, cols = data.shape
    x = data[:, 0: cols-1]
#     y = np.reshape((data[:, -1]), (rows, 1))
    y = data[:, -1]
    return x, y


# In[4]:


def n_fold_split(dataset, folds):
#     folds = 10
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


# In[5]:


def z_score_normalize_trainset(data):
    x, y = feature_result_split(data)
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    X = (x - means) / stds
#     return np.concatenate((X, y), axis=1), means, stds
    return X, y, means, stds


# In[6]:


def z_score_normalize_testset(data, means, stds):
    x, y = feature_result_split(data)
    x = (x - means)/stds
#     return np.concatenate((x, y), axis=1)
    return x, y


# In[7]:


def score_model(true, pred):
    true = true.astype(int)
    pred = pred.astype(int)
    
    k = len(np.unique(true))
    result = np.zeros((k, k))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
    precision = result[1][1] / (result[1][1] + result[0][1])
    recall = result[1][1] / (result[1][1] + result[1][0])
    
    return precision, recall


# In[26]:


def calc_AUC(labels, prob):
    f = list(zip(prob, labels))
    rank = [values2 for values1,values2 in sorted(f,key=lambda x:x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if(labels[i]==1):
            posNum+=1
        else:
            negNum+=1
    auc = 0
    auc = (sum(rankList)- (posNum*(posNum+1))/2)/(posNum*negNum)
    return auc


# In[8]:


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, threshold = calc_roc_curve(y_true, y_prob)
    
    roc_auc = calc_AUC(y_true, y_prob)
    
    lw = 2
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
 
    plt.show()


# In[9]:


def calc_roc_curve(y_true, y_prob):
    f = list(zip(y_prob, y_true))
    rank = [v2 for v1, v2 in sorted(f, key = lambda x: x[0])]
    unique_threshold = set(y_prob)
    unique_threshold.add(0)
    unique_threshold.add(1)
    thresholds = list(sorted(unique_threshold))
    fpr = list()
    tpr = list()
    for threshold in thresholds:
        y_pred = [1 if p >= threshold else 0 for p in y_prob]
        tp = 0
        p = 0
        fp = 0
        n = 0
        for i in range(len(y_pred)):
            if y_true[i] == 1:
                if y_pred[i] == 1:
                    tp += 1
                p += 1
            else:
                if y_pred[i] == 1:
                    fp += 1
                n += 1
        fpr.append(fp / n)
        tpr.append(tp / p)
    return fpr, tpr, thresholds


# In[23]:


def linear_kernel(df, K, M, C):
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_confusion_matrix = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_confusion_matrix = []
    i = 1
    
    for train, test in n_fold_split(df, K):
        print('*' * 60)
        print('Fold', i)
        org_train_data = np.take(df, train, 0).to_numpy()
        org_test_data = np.take(df, test, 0).to_numpy()
        best_score = 0
        best_c = 0
        
        for train_, test_ in n_fold_split(org_train_data, M):
            inner_train_data = np.take(org_train_data, train_, 0)
            inner_test_data = np.take(org_train_data, test_, 0)
            train_X, train_Y, scaled_mean, scaled_stds = z_score_normalize_trainset(inner_train_data)
            test_X, test_Y = z_score_normalize_testset(inner_test_data, scaled_mean, scaled_stds)
            for c in C:
                clf = SVC(C=c, kernel = 'linear')
                clf.probability = True
                clf.fit(train_X, train_Y)
                score = clf.score(test_X, test_Y)
                if(score > best_score):
                    best_score = score
                    best_c = c
        print('\nBest value of C:', best_c)
        scal_org_train_X, scal_org_train_Y, scal_mean, scal_stds = z_score_normalize_trainset(org_train_data)
        scal_org_test_X, test_Y = z_score_normalize_testset(org_test_data, scal_mean, scal_stds)
        outer_clf = SVC(C=best_c, kernel='linear')
        outer_clf.probability = True
        outer_clf.fit(scal_org_train_X, scal_org_train_Y)
        
        # training set
        print('Training set')
        pred_train_y = outer_clf.predict(scal_org_train_X)
        t_precision, t_recall = score_model(scal_org_train_Y, pred_train_y)
        train_accuracies.append(outer_clf.score(scal_org_train_X, scal_org_train_Y))
        train_precisions.append(t_precision)
        train_recalls.append(t_recall)
        print('\tAccuracy:', outer_clf.score(scal_org_train_X, scal_org_train_Y))
        print('\tPrecision:', t_precision)
        print('\tRecall:', t_recall)
        
        # test set
        print('Test set')
        pred_test_y = outer_clf.predict(scal_org_test_X)
        pred_test_prob = outer_clf.predict_proba(scal_org_test_X)
        
        precision, recall = score_model(test_Y, pred_test_y)
        test_accuracies.append(outer_clf.score(scal_org_test_X, test_Y))
        test_precisions.append(precision)
        test_recalls.append(recall)
        print('\tAccuracy:', outer_clf.score(scal_org_test_X, test_Y))
        print('\tPrecision:', precision)
        print('\tRecall:', recall)
        
        # plot roc
        pred_test_prob = [p[1] for p in pred_test_prob]
        plot_roc_curve(test_Y, pred_test_prob)
        
        i += 1
    print('\nTraining Data:')
    print('Average accuracy: ', np.mean(train_accuracies), '\tStandard deviation:', np.std(train_accuracies))
    print('Average precision: ', np.mean(train_precisions), '\tStandard deviation:', np.std(train_precisions)) 
    print('Average recall: ', np.mean(train_recalls), '\tStandard deviation:', np.std(train_recalls))
    print('\nTest Data:')
    print('Average accuracy: ', np.mean(test_accuracies), '\tStandard deviation:', np.std(test_accuracies))
    print('Average precision: ', np.mean(test_precisions), '\tStandard deviation:', np.std(test_precisions)) 
    print('Average recall: ', np.mean(test_recalls), '\tStandard deviation:', np.std(test_recalls))


# In[24]:


def RBF_kernel(df, K, M, C):
    gamma = [2**i for i in range(-5, 5)]
    
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_confusion_matrix = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_confusion_matrix = []
    i = 1
    
    for train, test in n_fold_split(df, K):
        print('*' * 60)
        print('Fold', i)
        org_train_data = np.take(df, train, 0).to_numpy()
        org_test_data = np.take(df, test, 0).to_numpy()
        best_score = 0
        best_c = 0
        best_gamma = 0
        
        for train_, test_ in n_fold_split(org_train_data, M):
            inner_train_data = np.take(org_train_data, train_, 0)
            inner_test_data = np.take(org_train_data, test_, 0)
            
            train_X, train_Y, scaled_mean, scaled_stds = z_score_normalize_trainset(inner_train_data)
            test_X, test_Y = z_score_normalize_testset(inner_test_data, scaled_mean, scaled_stds)
            
            for g in gamma:
                for c in C:
#                     print(c, g)
                    clf = SVC(C=c, kernel = 'rbf', gamma=g)
                    clf.probability = True
                    clf.fit(train_X, train_Y)
                    score = clf.score(test_X, test_Y)
                    if(score > best_score):
                        best_score = score
                        best_c = c
                        best_gamma = g
        print('\nBest value of C:', best_c)
        print('Best value of gamma:', best_gamma)
        scal_org_train_X, scal_org_train_Y, scal_mean, scal_stds = z_score_normalize_trainset(org_train_data)
        scal_org_test_X, test_Y = z_score_normalize_testset(org_test_data, scal_mean, scal_stds)
        outer_clf = SVC(C=best_c, kernel='rbf', gamma=best_gamma)
        outer_clf.probability = True
        outer_clf.fit(scal_org_train_X, scal_org_train_Y)
        
        # training set
        print('Training set')
        pred_train_y = outer_clf.predict(scal_org_train_X)
        t_precision, t_recall = score_model(scal_org_train_Y, pred_train_y)
        train_accuracies.append(outer_clf.score(scal_org_train_X, scal_org_train_Y))
        train_precisions.append(t_precision)
        train_recalls.append(t_recall)
        print('\tAccuracy:', outer_clf.score(scal_org_train_X, scal_org_train_Y))
        print('\tPrecision:', t_precision)
        print('\tRecall:', t_recall)
        
        # test set
        print('Test set')
        pred_test_y = outer_clf.predict(scal_org_test_X)
        pred_test_prob = outer_clf.predict_proba(scal_org_test_X)
        
        precision, recall = score_model(test_Y, pred_test_y)
        test_accuracies.append(outer_clf.score(scal_org_test_X, test_Y))
        test_precisions.append(precision)
        test_recalls.append(recall)
        print('\tAccuracy:', outer_clf.score(scal_org_test_X, test_Y))
        print('\tPrecision:', precision)
        print('\tRecall:', recall)
        
        # plot roc
        pred_test_prob = [p[1] for p in pred_test_prob]
        plot_roc_curve(test_Y, pred_test_prob)
        
        i += 1
    print('\nTraining Data:')
    print('Average accuracy: ', np.mean(train_accuracies), '\tStandard deviation:', np.std(train_accuracies))
    print('Average precision: ', np.mean(train_precisions), '\tStandard deviation:', np.std(train_precisions)) 
    print('Average recall: ', np.mean(train_recalls), '\tStandard deviation:', np.std(train_recalls))
    print('\nTest Data:')
    print('Average accuracy: ', np.mean(test_accuracies), '\tStandard deviation:', np.std(test_accuracies))
    print('Average precision: ', np.mean(test_precisions), '\tStandard deviation:', np.std(test_precisions)) 
    print('Average recall: ', np.mean(test_recalls), '\tStandard deviation:', np.std(test_recalls))


# In[21]:


def main(dataset, kernel):
    if dataset == "Spambase":
        df, lRate, tolerance, n_iters = loadSpambaseDataset()
        C = [2**i for i in range(-5, 5)]
    elif dataset == "Diabete":
        df, lRate, tolerance, n_iters = loadDiabetesDataset()
        C = [2**i for i in range(-5, 5)]
    elif dataset == "BreastCancer":
        df, lRate, tolerance, n_iters = loadBreastCancerDataset()
        C = [2**i for i in range(-5, 5)]
    else:
        print('Please input correct dataset name.')
        return
    
    K = 10
    M = 5
    
    
    if kernel == "linear":
        linear_kernel(df, K, M, C)
    elif kernel == "rbf":
        RBF_kernel(df, K, M, C)
    else:
        print('Please input correct kernel name.')
        return


# In[38]:


import sys
if __name__ == '__main__':
    dataset = sys.argv[1]
    kernel = sys.argv[2]
    main(dataset, kernel)


# In[ ]:




