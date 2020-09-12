#!/usr/bin/env python
# coding: utf-8

# In[24]:


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


class Decision_Node_MultiWay:
    def __init__(self, question, impurity):
        self.question = question
        self.children = []
        self.impurity = impurity
        
    def add_child(self, child_node):
        self.children.append(child_node)
        
    def get_child(self, idx):
        return self.children[idx] if idx < len(self.children) else 0
        


# In[4]:


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


# In[5]:


class Question_Multiway:
    def __init__(self, column, value_list):
        self.column = column
        self.values = {k: v for v, k in enumerate(value_list)}
        
    def match(self, example):
        val = example[self.column]
        return self.values.get(val, -1)
    
    def value_num(self):
        return len(self.values)
        
    def __repr__(self):
        return "value of col-%s?" % self.column


# In[6]:


def buildTree_multiway(datas, splitNum):
    gain, question, impurity = find_best_split_multiway(datas)
    if gain == 0 or len(datas) <= splitNum:
        return Leaf(datas, impurity)
    row_list = partition_multiway(datas, question)
    new_node = Decision_Node_MultiWay(question, impurity)
    for rows in row_list:
        new_node.add_child(buildTree_multiway(rows, splitNum))
    return new_node


# In[7]:


def partition_multiway(rows, question):
    row_list = [[] for i in range(question.value_num())]
    for row in rows:
        node_idx = question.match(row)
        if node_idx != -1:
            row_list[node_idx].append(row)
    
    return row_list


# In[8]:


def classify_multiway(row, node):
    if isinstance(node, Leaf):
        return node.type
    node_idx = node.question.match(row)
    if node_idx == -1:
        return None
    return classify_multiway(row, node.get_child(node_idx))


# In[9]:


def classCounts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# In[10]:


def find_best_split_multiway(datas):
    bestGain = 0
    bestQuestion = None
    currentImpurity = calc_entropy(datas)
    nFeatures = len(datas[0]) - 1
    
    for col in range(nFeatures):
        values = set([data[col] for data in datas])
        question = Question_Multiway(col, values)
        row_list = partition_multiway(datas, question)
        gain = infoGainMultiWay(row_list, currentImpurity)
        if gain >= bestGain:
            bestGain = gain
            bestQuestion = question
    return bestGain, bestQuestion, currentImpurity


# In[11]:


def calc_entropy(rows):
    counts = classCounts(rows)
    impurity = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity += -(prob_of_lbl * np.log2(prob_of_lbl))
    return impurity


# In[12]:


def infoGainMultiWay(row_list, entropy):
    len_sum = 0
    for row in row_list:
        len_sum += len(row)
    res = entropy
    for idx, row in enumerate(row_list):
        res -= (float(len(row_list[idx])) / len_sum) * calc_entropy(row_list[idx])
    return res


# In[13]:


def printLeaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


# In[14]:


def printTreeMultiway(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions, ", ", "Impurity ", node.impurity)
        return
    
    print(spacing + str(node.question))
    for idx, child in enumerate(node.children):
        print(spacing + '--> %s:' % idx)
        printTreeMultiway(child, spacing + "   ")


# In[32]:


def predict_multiway(rows, tree):
    accuracies = []
    results = []
    c_labels = np.unique(rows[:, -1])
#     print("=" * 100)
    for row in rows:
        res = classify_multiway(row, tree);
        results.append(res)
#         print("Actual result: ", row[-1], " Predicted: ", printLeaf(res))
        key = row[-1]
        if res != None and key == res:
            ac = 1.0
        else:
            ac = 0.0
        accuracies.append(ac)
    y_test = rows[:, -1]
    y_pred = np.array(results)
    cmtx = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=c_labels),
                        index=c_labels, columns=c_labels)
    stdeviation = statistics.stdev(accuracies)
#     print("Test accuracy: ", accuracies)
#     print("Average accuracy: ", statistics.mean(accuracies))
#     print("Standard deviation: ", str(stdeviation))
#     print("\n")
    return accuracies, stdeviation, cmtx     


# In[41]:


def main(threshold):
    filename = "mushroom.csv"
    df = pd.read_csv(filename, header=None);
    df_array = df.values
    kfold = KFold(n_splits=10, shuffle=True)
    accuracies = []
    deviations = []
    confusion_matrics = []
    cm_idx = ["e", "p"]
    cnt = 1
    print("Dataset: Mushroom (Multiway)\t\tSize of dataset: ", df.shape, "\t\tThreshold: ", threshold)
#     print("\n")
    for train, test in kfold.split(df):
        train_datas = np.take(df, train, 0).to_numpy()
        numOfDatas = len(train_datas)
        my_tree = buildTree_multiway(train_datas, threshold * numOfDatas)
#         printTreeMultiway(my_tree)
        test_datas = np.take(df, test, 0).to_numpy()
        lenOfTestDatas = len(test_datas)
        ac, dv, cm = predict_multiway(test_datas, my_tree)
        accuracies.append(sum(ac)/lenOfTestDatas)
        deviations.append(dv)
        confusion_matrics.append(cm.to_numpy())
        cnt += 1
    cm = pd.DataFrame(sum([c for c in confusion_matrics]),index=cm_idx, columns=cm_idx)
    print("-" * 100)
    print("Average accuracy of ten-fold cross validation: ", sum(accuracies) / len(accuracies))
    print("Average standard deviation of ten-fold cross validation: ", sum(deviations)/len(deviations))
    print("Confusion matrix: \n", cm)
    print("-" * 100)


# In[ ]:


import sys
if __name__ == "__main__":
    try:
        threshold = sys.argv[1]
    except:
        print('Please input threshold.')
        exit()
    main(float(threshold))


# In[ ]:




