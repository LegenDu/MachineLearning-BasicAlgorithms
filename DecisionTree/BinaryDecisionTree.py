#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import math
import statistics


# In[2]:


class Decision_Node:
    def __init__(self, question, true_branch, false_branch, impurity):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.impurity = impurity


# In[3]:


class Leaf:
    def __init__(self, datas, impurity):
        self.predictions = classCounts(datas)
        self.impurity = impurity
        curType = ""
        curInstance = 0
        for p in self.predictions:
            ins = self.predictions[p]
            if ins > curInstance:
                curInstance = ins
                curType = p
        self.type = curType


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
    gain, question, impurity = find_best_split(datas)
    if gain == 0 or len(datas) <= splitNum:
        return Leaf(datas, impurity)
    true_rows, false_rows = partition(datas, question)
    true_branch = buildTree(true_rows, splitNum)
    false_branch = buildTree(false_rows, splitNum)
    return Decision_Node(question, true_branch, false_branch, impurity)


# In[6]:


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


# In[7]:


def classify(row, node):
    if isinstance(node, Leaf):
        return node.type
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


# In[8]:


def classCounts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# In[9]:


def find_best_split(datas):
    bestGain = 0
    bestQuestion = None
    currentImpurity = calc_entropy(datas)
    nFeatures = len(datas[0]) - 1
    
    for col in range(nFeatures):
        values = set([data[col] for data in datas])
        for val in values:
            question = Question(col, val)
            trueRows, falseRows = partition(datas, question)
            if len(trueRows) == 0 or len(falseRows) == 0:
                continue
            gain = infoGain(trueRows, falseRows, currentImpurity)
            if gain >= bestGain:
                bestGain = gain
                bestQuestion = question
    return bestGain, bestQuestion, currentImpurity


# In[10]:


def calc_entropy(rows):
    counts = classCounts(rows)
    impurity = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity += -(prob_of_lbl * np.log2(prob_of_lbl))
    return impurity


# In[11]:


def infoGain(left, right, entropy):
    p = float(len(left)) / (len(left) + len(right))
    return entropy - p * calc_entropy(left) - (1 - p) * calc_entropy(right)


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
        print(spacing + "Predict", node.predictions, ", ", "Type ", node.type)
        return
    print(spacing + str(node.question))
    print(spacing + '--> True:')
    printTree(node.true_branch, spacing + "   ")
    print(spacing + '--> False:')
    printTree(node.false_branch, spacing + "   ")


# In[14]:


def predict(rows, tree):
    accuracies = []
    keys = []
    results = []
    c_labels = np.unique(rows[:, -1])
    for row in rows:
        res = classify(row, tree);
        results.append(res)
#         print("Actual result: ", row[-1], " Predicted: ", res)
        key = row[-1]
        
        if key == res:
            ac = 1.0
        else:
            ac = 0.0
        accuracies.append(ac)
    y_test = rows[:, -1]
    y_pred = np.array(results)
    
    cmtx = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=c_labels),
                        index=c_labels, columns=c_labels)
    stdeviation = statistics.stdev(accuracies)
    return accuracies, stdeviation, cmtx     


# In[104]:


def loadIrisDataset():
    filename = "iris.csv"
    datafile = pd.read_csv(filename, header=None);
    dataset = "Iris"
    df_array = datafile.values
    X = df_array[:, 0:4]
    Y = df_array[:, 4]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(X)
    df = pd.DataFrame(x_scaled)
    df[4] = Y
    return df

def loadSpambaseDataset():
    filename = "spambase.csv"
    dataset = "Spambase"
    datafile = pd.read_csv(filename, header=None, delimiter=",")
    df_array = datafile.values
    X = df_array[:, 0:57]
    Y = df_array[:, 57]
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(X)
    df = pd.DataFrame(x_scaled)
    df[57] = Y
    return df

def loadMushroomDataset():
    filename = "mushroom.csv"
    dataset = "Mushroom (Binary Split)"
    df = pd.read_csv(filename, header=None);
    return df


# In[110]:


def main(dataset, threshold):
    df = None
    if dataset == "Iris":
        df = loadIrisDataset()
    elif dataset == "Spambase":
        df = loadSpambaseDataset()
    elif dataset == "Mushroom":
        df = loadMushroomDataset()
    else:
        print("Incorrect dataset name!")
        return
        
    kfold = KFold(n_splits=10, shuffle=True)
    accuracies = []
    deviations = []
    confusion_matrics = []
    cm_rows = []
    cm_cols = []
    cnt = 1
    print("Dataset: ", dataset, "\t\tSize of dataset: ", df.shape, "\t\tThreshold: ", threshold)
    # print("\n")
    for train, test in kfold.split(df):
        train_datas = np.take(df, train, 0).to_numpy()
        numOfDatas = len(train_datas)
        my_tree = buildTree(train_datas, threshold * numOfDatas)
    #     printTree(my_tree)
        test_datas = np.take(df, test, 0).to_numpy()
        lenOfTestDatas = len(test_datas)
    #     print("Test ", cnt, "\tNumber of test data: ", lenOfTestDatas)
        ac, dv, cmtx = predict(test_datas, my_tree)
        accuracies.append(sum(ac)/lenOfTestDatas)
        deviations.append(dv)
        confusion_matrics.append(cmtx.to_numpy())
        cm_rows = cmtx.index
        cm_cols = cmtx.columns
        cnt += 1
    cm = pd.DataFrame(sum([c for c in confusion_matrics]),
                            index=cm_rows, columns=cm_cols)

    print("-" * 100)
    print("Average accuracy of ten-fold cross validation: ", sum(accuracies) / len(accuracies))
    print("Average standard deviation of ten-fold cross validation: ", sum(deviations)/len(deviations))
    print("Confusion matrix: \n", cm)
    print("-" * 100)


# In[112]:


import sys
if __name__ == '__main__':
    dataset = sys.argv[1]
    threshold = float(sys.argv[2])
    main(dataset, threshold)


# In[ ]:




