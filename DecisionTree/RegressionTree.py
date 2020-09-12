#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import preprocessing
import math
import sys
import statistics


# In[2]:


class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


# In[3]:


class Leaf:
    def __init__(self, datas, value):
        self.predictions = classCounts(datas)
        self.value = value


# In[4]:


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
        
    def match(self, example):
        val = example[self.column]
        if isinstance(val, int) or isinstance(val, float):
            return val >= self.value
        else:
            return val == self.value
        
    def __repr__(self):
        condition = "=="
        if isinstance(self.value, int) or isinstance(self.value, float):
            condition = ">="
        return "Is col-%s %s %s?" % (self.column, condition, str(self.value))


# In[5]:


def buildTree(datas, splitNum):
    sse, question = find_best_split(datas)
    if sse == 0 or len(datas) <= splitNum:
        d = np.array(datas)
        return Leaf(datas, np.mean(d[:, -1]))
    true_rows, false_rows = partition(datas, question)
    true_branch = buildTree(true_rows, splitNum)
    false_branch = buildTree(false_rows, splitNum)
    return Decision_Node(question, true_branch, false_branch)


# In[6]:


def partition(rows, question):
    true_rows, false_rows = [],[]
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


# In[7]:


def classify(row, node):
    if isinstance(node, Leaf):
        return node.value
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


# In[8]:


def find_best_split(datas):
    bestSSE = sys.maxsize
    bestQuestion = None
    nFeatures = len(datas[0]) - 1
    
    for col in range(nFeatures):
        values = set([data[col] for data in datas])
        for val in values:
            question = Question(col, val)
            trueRows, falseRows = partition(datas, question)
            if len(trueRows) == 0 or len(falseRows) == 0:
                continue
            sse = sum_of_squared_error(trueRows, falseRows)
            if sse <= bestSSE:
                bestSSE = sse
                bestQuestion = question
    return bestSSE, bestQuestion


# In[9]:


def calc_squared_error(rows, mean):
    se = 0;
    for row in rows:
        value = row[-1]
        se += (value - mean)**2
    return se


# In[10]:


def sum_of_squared_error(left, right):
    left = np.array(left)
    right = np.array(right)
    l_mean = np.mean(left[:, -1])
    r_mean = np.mean(right[:, -1])
    return calc_squared_error(left, l_mean) + calc_squared_error(right, r_mean)


# In[11]:


def classCounts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# In[12]:


def printLeaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


# In[13]:


def printTree(node, spacing=""):
    if isinstance(node, Leaf):
#         print(spacing + "Predict", node.predictions, ", ", node.value)
        print(spacing + "Predict", node.value)
        return
    print(spacing + str(node.question))
    print(spacing + '--> True:')
    printTree(node.true_branch, spacing + "   ")
    print(spacing + '--> False:')
    printTree(node.false_branch, spacing + "   ")


# In[17]:


def predict(rows, tree):
    sses = []
#     print("=" * 100)
    for row in rows:
        res = classify(row, tree);
#         print("Actual result: ", row[-1], " Predicted: ", printLeaf(res))
        key = row[-1]
        sse = (key - res)**2
        sses.append(sse)
    stdeviation = statistics.stdev(sses)
#     print("Test sum of squeared error: ", sses)
#     print("Average sse: ", statistics.mean(sses))
#     print("Standard deviation: ", str(stdeviation))
#     print("\n")
    return sses, stdeviation     


# In[20]:


def main(threshold):  
    # Housing
    filename = "housing.csv"
    # datafile = pd.read_csv(filename, header=None);
    datafile = pd.read_csv(filename, header=None)
    df_array = datafile.values
    X = df_array[:, 0:13]
    Y = df_array[:, 13]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(X)
    df = pd.DataFrame(x_scaled)
    df[13] = Y
    kfold = KFold(n_splits=10, shuffle=True) 
    sses = []
    deviations = []
    cnt = 1
    print("Dataset: Housing\t\tSize of dataset: ", df.shape, "\t\tThreshold: ", threshold)
    for train, test in kfold.split(df):
        train_datas = np.take(df, train, 0).to_numpy()
        numOfDatas = len(train_datas)
        my_tree = buildTree(train_datas, threshold * numOfDatas)
    #     printTree(my_tree)
        test_datas = np.take(df, test, 0).to_numpy()
        lenOfTestDatas = len(test_datas)
    #     print("Test ", cnt, "\tNumber of test data: ", lenOfTestDatas)
        sse, dv = predict(test_datas, my_tree)
        sses.append(sum(sse)/lenOfTestDatas)
        deviations.append(dv)
        cnt += 1
    print("-" * 100)
    print("Average sum of squared error of ten-fold cross validation: ", sum(sses) / len(sses))
    print("Average standard deviation of ten-fold cross validation: ", sum(deviations)/len(deviations))
    print("-" * 100)


# In[ ]:

import sys
if __name__ == "__main__":
    threshold = float(sys.argv[1])
    main(threshold)

