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


def loadDigitsDataset():
    train_filename ="Datasets/digits train.csv"
    test_filename = "Datasets/digits test.csv"
    max_iters = 300
    return pd.read_csv(train_filename, header=None, delimiter=","), pd.read_csv(test_filename, header=None), max_iters


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


# In[120]:


def feature_result_split(data):
    rows, cols = data.shape
    x = data[:, 0: -1]
    y = data[:, -1]
    return x, y


# In[82]:


def score_model(true, pred):
    true = true.astype(int)
    
    k = len(np.unique(true))
    result = np.zeros((k, k))
    for i in range(len(true)):
        result[true[i]][pred[i]] += 1
#     print(result)
    ac = 0
    precision = []
    recall = []
    for i in range(k):
        pre = 0
        rec = 0
        for j in range(k):
            pre += result[j][i]
            rec += result[i][j]
        ac += result[i][i]
        precision.append(result[i][i] / pre)
        recall.append(result[i][i] / rec)
    accuracy = ac / len(true)
    
    return accuracy, precision, recall


# In[83]:


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


# In[84]:


def plot_roc_curve(y_true, y_prob):
    color = ['darkorange', 'blue', 'green']
    for i in range(len(y_prob)):
        y_label = encodeLabels(y_true, i)
        fpr, tpr, threshold = calc_roc_curve(y_label, y_prob[i])

        roc_auc = calc_AUC(y_label, y_prob[i])

        lw = 2
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, color=color[i % 3],
                 lw=lw, label='ROC curve for class %i (area = %0.3f)' % (i, roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        plt.show()


# In[85]:


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


# In[86]:


def encodeLabels(labels, cls):
    return [1 if i == cls else 0 for i in labels]


# In[87]:


def predict_rbf(train_X, train_Y, test_data, class_list, best_c, best_gamma):
    k = len(class_list)
    y_pred = []
    for j in range(k):
        positive_class = class_list[j]
        tmp_train_Y = encodeLabels(train_Y, positive_class)
        clf = SVC(C=best_c[j], kernel = 'rbf', gamma=best_gamma[j])
        clf.probability = True
        clf.fit(train_X, tmp_train_Y)
        p = clf.predict_proba(test_data)[:,1]
        y_pred.append(p)
    
    pred = predict_y(len(test_data), k, y_pred)    
    return pred, y_pred


# In[88]:


def predict_y(n, class_num, y_pred):
    pred = []
    for i in range(n):
        cls = 0
        prob = 0
        for j in range(class_num):
            if y_pred[j][i] > prob or prob == 0:
                cls = j
                prob = y_pred[j][i]
        pred.append(cls)
    return pred


# In[89]:


def predict_linear(train_X, train_Y, test_data, class_list, best_c):
    k = len(class_list)
    y_pred = []

    for j in range(len(class_list)):
        positive_class = class_list[j]
        tmp_train_Y = encodeLabels(train_Y, positive_class)
        clf = SVC(C=best_c[j], kernel = 'linear')
        clf.probability = True
        clf.fit(train_X, tmp_train_Y)
        p = clf.predict_proba(test_data)[:,1]
        y_pred.append(p)
    
    pred = predict_y(len(test_data), k, y_pred)
    
    return pred, y_pred


# In[90]:


def linear_kernel(train, test, M, C):
    org_train_data = train.to_numpy()
    org_test_data = test.to_numpy()
    class_list = np.unique(org_train_data[:, -1])

    best_score = [0 for i in range(len(class_list))]
    best_c = [0 for i in range(len(class_list))]
        
    for train_, test_ in n_fold_split(org_train_data, M):
        inner_train_data = np.take(org_train_data, train_, 0)
        inner_test_data = np.take(org_train_data, test_, 0)
        
        train_X, train_Y = feature_result_split(inner_train_data)
        test_X, test_Y = feature_result_split(inner_test_data)
        
        for i in range(len(class_list)):
            positive_class = class_list[i]
            tmp_train_Y = encodeLabels(train_Y, positive_class)
            tmp_test_Y = encodeLabels(test_Y, positive_class)
            
            for c in C:
                clf = SVC(C=c, kernel = 'linear')
                clf.fit(train_X, tmp_train_Y)
                score = clf.score(test_X, tmp_test_Y)
                if score > best_score[i] or best_score[i] == 0:
                    best_score[i] = score
                    best_c[i] = c
            
    print('\nBest value of C:', best_c)
    
    train_X, train_Y = feature_result_split(org_train_data)
    test_X, test_Y = feature_result_split(org_test_data)
    
    # Training set
    pred_train_y, prob_train_y = predict_linear(train_X, train_Y, train_X, class_list, best_c)
    t_accuracy, t_precision, t_recall = score_model(train_Y, pred_train_y)
    print('Training set\nAccuracy:', t_accuracy, '\nPrecision:', t_precision, '\nRecall:', t_recall)
    print('\n')
    # test set
    pred_test_y, prob_test_y = predict_linear(train_X, train_Y, test_X, class_list, best_c)
    accuracy, precision, recall = score_model(test_Y, pred_test_y)
    print('Test set:\nAccuracy:', accuracy, '\nPrecision:', precision, '\nRecall:', recall)

    plot_roc_curve(test_Y, prob_test_y)


# In[91]:


def RBF_kernel(train, test, M, C):
    gamma = [2**i for i in range(-10, 0)]
    org_train_data = train.to_numpy()
    org_test_data = test.to_numpy()
        
    class_list = np.unique(org_train_data[:, -1])
    best_score = [0 for i in range(len(class_list))]
    best_c = [0 for i in range(len(class_list))]
    best_gamma = [0 for i in range(len(class_list))]
        
    for train_, test_ in n_fold_split(org_train_data, M):
        inner_train_data = np.take(org_train_data, train_, 0)
        inner_test_data = np.take(org_train_data, test_, 0)

        train_X, train_Y = feature_result_split(inner_train_data)
        test_X, test_Y = feature_result_split(inner_test_data)

        for i in range(len(class_list)):
            positive_class = class_list[i]
            tmp_train_Y = encodeLabels(train_Y, positive_class)
            tmp_test_Y = encodeLabels(test_Y, positive_class)
            
            for g in gamma:
                for c in C:
                    clf = SVC(C=c, kernel = 'rbf', gamma=g)
                    clf.probability = True
                    clf.fit(train_X, train_Y)
                    score = clf.score(test_X, test_Y)
                    if score > best_score[i] or best_score[i] == 0:
                        best_score[i] = score
                        best_c[i] = c
                        best_gamma[i] = g

    print('\nBest value of C:', best_c)
    print('Best value of gamma:', best_gamma)
    
    train_X, train_Y = feature_result_split(org_train_data)
    test_X, test_Y = feature_result_split(org_test_data)

    # training set
    pred_train_y, prob_train_Y = predict_rbf(train_X, train_Y, train_X, class_list, best_c, best_gamma)
    t_accuracy, t_precision, t_recall = score_model(train_Y, pred_train_y)
    print('Training set:\nAccuracy:', t_accuracy, '\nPrecision:', t_precision, '\nRecall:', t_recall)
    print('\n')
    # test set
    pred_test_y, prob_test_y = predict_rbf(train_X, train_Y, test_X, class_list, best_c, best_gamma)

    accuracy, precision, recall = score_model(test_Y, pred_test_y)
    print('Test set:\nAccuracy:', accuracy, '\nPrecision:', precision, '\nRecall:', recall)

    # plot roc
    plot_roc_curve(test_Y, prob_test_y)


# In[92]:


def main(kernel):
    C = [2**i for i in range(-10, 0)]
    M = 5
    
    df_train, df_test, n_iters = loadDigitsDataset()
    
    if kernel == "linear":
        linear_kernel(df_train, df_test, M, C)
    elif kernel == "rbf":
        RBF_kernel(df_train, df_test, M, C)
    else:
        print('Please input correct kernel name.')
        return


# In[96]:


import sys
if __name__ == '__main__':
    kernel = sys.argv[1]
    main(kernel)


# In[ ]:




