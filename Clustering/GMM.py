#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import random
import sys
np.random.seed(2)


# In[2]:


def loadDermatologyDataset():
    filename ="Datasets/dermatologyData.csv"
    return pd.read_csv(filename, header=None).to_numpy()

def loadVowelDataset():
    filename ="Datasets/vowelsData.csv"
    return pd.read_csv(filename, header=None).to_numpy()

def loadGlassDataset():
    filename ="Datasets/glassData.csv"
    return pd.read_csv(filename, header=None).to_numpy()

def loadEcoliDataset():
    filename ="Datasets/ecoliData.csv"
    return pd.read_csv(filename, header=None).to_numpy()

def loadYeastDataset():
    filename ="Datasets/yeastData.csv"
    return pd.read_csv(filename, header=None).to_numpy()

def loadSoybeanDataset():
    filename ="Datasets/soybeanData.csv"
    return pd.read_csv(filename, header=None).to_numpy()


# In[3]:


def dist(x1, x2):
    return np.linalg.norm(x1 - x2)


# In[4]:


def calc_loglikelihood(X, K, Gamma, Alpha):
    ll = 0
    for i in range(X.shape[0]):
        before = 0
        for j in range(K):
            before += Alpha[j] * Gamma[i][j]
        ll += np.log(before)
    return ll


# In[5]:


def initMiu(X, k):
    n_samples = X.shape[0]
    index = np.random.choice(n_samples, 1, replace=False)
    Mius = []
    Mius.append(X[index])
    
    for _  in range(k - 1):
        max_dist_index = 0
        max_distance = 0
        
        for j in range(n_samples):
            min_dist_with_miu = 999999
            
            for miu in Mius:
                dist_with_miu = dist(miu, X[j])
                if min_dist_with_miu > dist_with_miu:
                    min_dist_with_miu = dist_with_miu
            
            if max_distance < min_dist_with_miu:
                max_distance = min_dist_with_miu
                max_dist_index = j
        Mius.append(X[max_dist_index])
        
        
    mius_array = np.array([])
    for i in range(k):
        if i == 0:
            mius_array = Mius[i]
        else:
            Mius[i] = Mius[i].reshape(Mius[0].shape)
            mius_array = np.append(mius_array, Mius[i], axis=0)
    
    return mius_array


# In[6]:


def update_Miu(X, label, Gamma):
    a = 0
    b = 0
    for i in range(X.shape[0]):
        a += Gamma[i][label] * X[i]
        b += Gamma[i][label]
    if b == 0:
        b = 1e-10
    return a / b


# In[7]:


def update_Sigma(X, label, miu, Gamma):
    a = 0
    b = 0
    
    for i in range(X.shape[0]):
        X[i] = X[i].reshape(1, -1)
        miu = miu.reshape(1, -1)
        a += Gamma[i][label] * (X[i] - miu).T * (X[i] - miu)
        b += Gamma[i][label]
        
    if b == 0:
        b = 1e-10
    return a / b


# In[8]:


def update_Alpha(X, label, Gamma):
    a = 0
    for i in range(X.shape[0]):
        a += Gamma[i][label]
    return a / X.shape[0]


# In[9]:


def cluster_assign(Gamma, n_samples, K):
    clusters = [[] for _ in range(K)]
    y_label = np.zeros([n_samples])
    for i in range(n_samples):
        type_y = -1
        tmp_gamma = -1
        for j in range(K):
            if tmp_gamma < Gamma[i][j]:
                tmp_gamma = Gamma[i][j]
                type_y = j
        y_label[i] = type_y
        clusters[type_y].append(i)

    return clusters, y_label


# In[10]:


def calc_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# In[11]:


def calc_sse(X, Mius, clusters):
    sse = 0
    for idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            sse += calc_distance(X[sample_idx], Mius[idx]) ** 2
    return sse


# In[12]:


def calc_entropy(prob):
    return -1 * prob * math.log(prob, 2)


# In[13]:


def calc_nmi(X, Y, clusters):
    H_Y = 0
    lbs, l_counts = np.unique(Y, return_counts=True)
    k = len(lbs)

    n_samples = X.shape[0]
    for c in l_counts:
        H_Y += calc_entropy(c / n_samples)
    
    H_C = 0
    for cluster in clusters:
        c_len = len(cluster)
        if c_len == 0: 
            continue
        H_C += calc_entropy(c_len / n_samples)
    
    
    H_YC = 0
    for cluster in clusters:
        c_len = len(cluster)
        data = Y[cluster]
        lbs, cnts = np.unique(data, return_counts=True)
        yc = 0
        for c in cnts:
            yc += calc_entropy(c / c_len)
        H_YC += yc * (c_len / n_samples)

    I_YC = H_Y - H_YC
    return (2 * I_YC) / (H_Y + H_C)


# In[14]:


def prob(x, mu, sigma):
    x = x.reshape(1, -1)
    mu = mu.reshape(1, -1)
    n = x.shape[1]

    covdet = np.linalg.det(sigma)
    
    # avoiding sigular matrix
    if covdet < 1e-5:
        covdet = np.linalg.det(sigma + np.eye(n) * 0.01)
        covinv = np.linalg.inv(sigma + np.eye(n) * 0.01)
    else:
        covinv = np.linalg.inv(sigma)

    expOn = -0.5 * (x - mu) @ (covinv) @ ((x - mu).T)
    divBy = np.float_power(2 * np.pi, n / 2) * np.float_power(covdet, 0.5) 
#     print('det', covdet, 'inv', covinv, np.exp(expOn) / divBy)
    return np.exp(expOn) / divBy


# In[15]:


def fit(X, Y, K, tolerance=0.01, n_iters=10):
    n_samples, n_features = X.shape
    Alpha = [1 / K for _ in range(K)]
    Mius = initMiu(X, K)
    Sigma = np.array([np.eye(n_features, dtype=float) * 0.1] * K)
    Gamma = np.zeros([n_samples, K])
    
    iter = 0
    ll = 0
    while iter < n_iters:
#         print(str(iter), '='*40)
        old_ll = ll
        for i in range(n_samples):
            sumAlphaMulP = 0
            for j in range(K):
                Gamma[i][j] = Alpha[j] * prob(X[i], Mius[j], Sigma[j])
                sumAlphaMulP += Gamma[i][j]
            
            for k in range(K):
                Gamma[i][k] /= sumAlphaMulP
        
        for i in range(K):
            Mius[i] = update_Miu(X, i, Gamma)
            Sigma[i] = update_Sigma(X, i, Mius[i], Gamma)
            Alpha[i] = update_Alpha(X, i, Gamma)
        
        ll = calc_loglikelihood(X, K, Gamma, Alpha)
        
        if abs(ll - old_ll) < tolerance:
            break
        iter += 1

    clusters, pred_y = cluster_assign(Gamma, n_samples, K)

    return clusters, pred_y, Mius


# In[60]:


def main(filename, k_pref):
    dataset = None
    if filename == 'Dermatology':
        dataset = loadDermatologyDataset()
        k_range = 10
    elif filename == 'Vowels':
        dataset = loadVowelDataset()
        k_range = 5
    elif filename == 'Glass':
        dataset = loadGlassDataset()
        k_range = 10
    elif filename == 'Ecoli':
        dataset = loadEcoliDataset()
        k_range = 5
    elif filename =='Yeast':
        dataset = loadYeastDataset()
        k_range = 4
    elif filename == 'Soybean':
        dataset = loadSoybeanDataset()
        k_range = 5
    else:
        print('Please input correct dataset name.')
        return

    data_X = dataset[:, 0: -1]
    data_Y = dataset[:, -1]
    classes = len(np.unique(data_Y))
    if k_pref == 'c':
        sse = float("nan")
        while math.isnan(sse):
            clusters, pred_y, Mius = fit(data_X, data_Y, classes, 1e-3, 20)
            sse = calc_sse(data_X, Mius, clusters)
        nmi = calc_nmi(data_X, data_Y, clusters)
        plt.figure()
        plt_name = filename + 'Dataset(GMM) K=' + str(classes)
        plt.title(plt_name)
        plt.scatter(data_X[:, 0], data_X[:, 1], marker='o', c=data_Y)
        plt.scatter(Mius[:, 0], Mius[:, 1], marker='^', c='r')
        plt.show()
        print('Num of Cluster\t\t\tSSE\t\t\tNMI')
        print(classes, '\t\t', sse, '\t', nmi)
    elif k_pref == 'l':
        sses = []
        nmis = []
        k_list = [i for i in range(2, classes + k_range)]
        if filename == 'Dermatology':
            k_list = [i for i in range(4, classes + k_range)]
        for K in k_list:
            sse = float("nan")
            while math.isnan(sse) or (len(sses) > 0 and sses[-1] < sse):
                clusters, pred_y, Mius = fit(data_X, data_Y, K, 0.01, 10)
                sse = calc_sse(data_X, Mius, clusters)
            sses.append(sse)
            nmi = calc_nmi(data_X, data_Y, clusters)
            nmis.append(nmi)
            print('Num of Cluster\t\t\tSSE\t\t\tNMI')
            print('\t', K, '\t\t', sse, '\t', nmi)
        plot_sse_vs_k(sses, k_list)
        plot_nmi_vs_k(nmis, k_list)
    else:
        print('Please input valid K preference.')
        return


# In[17]:


def plot_sse_vs_k(sses, k):
    plt.figure()
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.xticks(np.arange(min(k), max(k) + 1, 1.0))
    plt.plot(k, sses, 'g-')
    plt.show()


# In[18]:


def plot_nmi_vs_k(nmis, k):
    plt.figure()
    plt.xlabel('Number of Clusters')
    plt.ylabel('NMI')
    plt.xticks(np.arange(min(k), max(k) + 1, 1.0))
    plt.plot(k, nmis, 'g-')
    plt.show()


# In[19]:


if __name__ == '__main__':
    dataset = sys.argv[1]
    k_pref = sys.argv[2]
    main(dataset, k_pref)



