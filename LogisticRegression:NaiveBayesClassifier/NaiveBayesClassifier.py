#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from collections import OrderedDict
import collections

import matplotlib.pyplot as plt
filePath = "Datasets/20NG_data/"


# In[2]:


def loadTrainData():
    raw_data = pd.read_csv(filePath + "train_data.csv", sep=" ", header=None)
    train_data_label = pd.read_csv(filePath + "train_label.csv", sep=" ", header=None)
    return raw_data.to_numpy(), train_data_label.to_numpy()

def loadTestData():
    raw_data = pd.read_csv(filePath + "test_data.csv", sep=" ", header=None) 
    test_data_label = pd.read_csv(filePath + "test_label.csv", sep=" ", header=None)
    return raw_data.to_numpy(), test_data_label.to_numpy()


# In[3]:


def word_frequency_list(raw_data, K):
    word_freq = dict()
    for i in range(len(raw_data)):
        key = raw_data[i][1]
        count = raw_data[i][2]
        if key in word_freq.keys():
            word_freq[key] += count;
        else:
            word_freq[key] = count;
    word_freq_list = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)
    if K != 'All':
        res = [a[0] for a in word_freq_list[0: K]]
    else:
        res = [a[0] for a in word_freq_list]
    return res


# In[4]:


def filter_by_vocabulary(data, top_K_word_list):
    return data[np.where([i in top_K_word_list for i in data[:,1]])]


# In[5]:


def word_to_vector_bernoulliNB(data, vocabulary):
    sample = []
    cur = 0
    i = 0
    while i < len(data):
        x = [0] * len(vocabulary)
        cnt = i
        if(i == 0):
            cur = data[i][0]
        while cnt < len(data) and cur == data[cnt][0]:
            if data[cnt][1] in vocabulary:
                x[vocabulary.index(data[cnt][1])] = 1
            cnt += 1
        sample.append(x)
        if cnt >= len(data):
            break;
        cur = data[cnt][0]
        i = cnt
    return sample         


# In[6]:


def word_to_vector_multinomialNB(data, vocabulary):
    sample = []
    cur = 0
    i = 0
    while i < len(data):
        x = [0] * len(vocabulary)
        cnt = i
        if(i == 0):
            cur = data[i][0]
        while cnt < len(data) and cur == data[cnt][0]:
            if data[cnt][1] in vocabulary:
                x[vocabulary.index(data[cnt][1])] += data[cnt][2]
            cnt += 1
        sample.append(x)
        if cnt >= len(data):
            break;
        cur = data[cnt][0]
        i = cnt
    return sample   


# In[7]:


def calc_prob_bernoulliNB(X, Y, K):
    rows, columns = X.shape
    unique, l_counts = np.unique(Y, return_counts=True)
    pre_prob = l_counts / rows
    
    p_num = 2.0
    w_num = np.zeros((len(unique), 1)) + p_num
    w_freq = np.ones((len(unique), len(X[0])))
    for index, data in enumerate(X):
        label = Y[index]
        w_freq[label - 1] += data
        w_num[label - 1] += np.sum(data)
    w_prob = w_freq * 1.0 / (w_num + K)
    return pre_prob, w_prob


# In[8]:


def classify(X, d_prob, w_prob):
    test_class = np.zeros(len(X))

    for index, data in enumerate(X):
        pred = []
        c = 0
        for i in range(0, 20):
            p = np.sum(np.log(data * w_prob[i] + (1 - data) * (1 - w_prob[i]))) + np.log(d_prob[i])
            pred.append(p)
        m_pred = 0
        c_pred = 0
        for i in range(0, 20):
            if i == 0 or pred[i] > m_pred:
                m_pred = pred[i]
                c_pred = i + 1
        test_class[index] = c_pred
    return test_class


# In[19]:


def score_model(true, pred):
    pred = pred.astype(int)
    true = np.reshape(true, (1, len(true)))[0]
    confusion_matrix = np.zeros((21, 21))

    for i in range(len(true)):
        confusion_matrix[true[i]][pred[i]] += 1

    accuracy = []
    precision = []
    recall = []
    r_sum = 0
    n_sample = len(pred)
    
    false_positive = 0
    false_negative = 0
    for i in range(1, 21):
        true_positive = confusion_matrix[i][i]
        r_sum += true_positive
        false_positive = 0
        false_negative = 0
        total = 0
        for j in range(1, 21):
            total += confusion_matrix[i][j]
            if j != i:
                false_negative += confusion_matrix[i][j]
                false_positive += confusion_matrix[j][i]
        accuracy.append(true_positive / total)
        recall.append(true_positive / (true_positive + false_negative))
        precision.append(true_positive / (true_positive + false_positive))
    
#     print("Accuracy for each class:", accuracy)
#     print("Precision for each class:", precision)
#     print("Recall for each class:", recall)
    
    # print('confusion matrix:')
    # index = [i for i in range(1, 21)]
    # print(pd.DataFrame(data=confusion_matrix[1:,1:], index=index, columns=index))
    
    
    return r_sum / len(pred), accuracy, recall, precision


# In[10]:


def train_BernoulliModel(train_X, train_Y, vocabulary):
    train_X = np.asarray(word_to_vector_bernoulliNB(train_X, vocabulary))
    p_prob, w_prob = calc_prob_bernoulliNB(train_X, train_Y, len(vocabulary))
    return p_prob, w_prob    


# In[11]:


def train_MultinomialModel(train_X, train_Y, vocabulary):
    train_X = np.asarray(word_to_vector_multinomialNB(train_X, vocabulary))
    p_prob, w_prob = calc_prob_bernoulliNB(train_X, train_Y, len(vocabulary))
    return p_prob, w_prob


# In[32]:


def main(K):
    df, train_Y = loadTrainData()
    org_test_X, test_Y = loadTestData()
    vocabulary = word_frequency_list(df, K)
    test_X = np.asarray(word_to_vector_bernoulliNB(org_test_X, vocabulary))

    p_prob, w_prob = train_BernoulliModel(df, train_Y, vocabulary)
    pred_y = classify(test_X, p_prob, w_prob)
    b_model_accuracy, b_class_accuracy, b_recall, b_precision = score_model(test_Y, pred_y)
    print("Bernoulli Model (Vocabulary Size ", K, "):")
    print("accuracy:", b_model_accuracy, "\trecall: ", np.mean(b_recall), "\tprecision", np.mean(b_precision))
    print("\nAccuracy for each class:", b_class_accuracy)
    print("\nRecall for each class:", b_recall)
    print("\nPrecision for each class:", b_precision)
    
    p_prob, w_prob = train_MultinomialModel(df, train_Y, vocabulary)
    test_X = np.asarray(word_to_vector_bernoulliNB(org_test_X, vocabulary))
    pred_y = classify(test_X, p_prob, w_prob)
    m_model_accuracy, m_class_accuracy, m_recall, m_precision = score_model(test_Y, pred_y)
    print("\nMultinomial Event Model (Vocabulary Size ", K, "):")
    print("accuracy:", m_model_accuracy, "\trecall: ", np.mean(m_recall), "\tprecision", np.mean(m_precision))
    print("\nAccuracy for each class:", m_class_accuracy)
    print("\nRecall for each class:", m_recall)
    print("\nPrecision for each class:", m_precision)


# In[ ]:


import sys
if __name__ == '__main__':
    vocab_size = sys.argv[1]
    if vocab_size != "All":
        main(int(vocab_size))
    else:
        main(vocab_size)

