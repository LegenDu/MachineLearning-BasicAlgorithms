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


def calc_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# In[4]:


def kMeans(data_X, k, n_iters, tol=0):
    n_samples, n_features = data_X.shape
    
    # initialize centroids
    history_centroids = []
    centroids = data_X[np.random.randint(0, n_samples - 1, size=k)]
    history_centroids.append(centroids)
    centroids_old = np.zeros(centroids.shape)
    clusters = [[] for _ in range(k)]
    
    for _ in range(n_iters):
        clusters = create_clusters(centroids, data_X, k)
        centroids_old = centroids
        centroids = calc_centroids(data_X, clusters, k, n_samples, n_features)
        history_centroids.append(centroids)
        if calc_distance(centroids_old, centroids) == 0:
            break
    sse = calc_sse(data_X, centroids, clusters)
#     nmi = calc_nmi(data_X, centroids, clusters)
    return sse, clusters, centroids


# In[5]:


def create_clusters(centroids, X, k):
    clusters = [[] for _ in range(k)]
    for idx, sample in enumerate(X):
        centroid_idx = closest_centroid(sample, centroids)
        clusters[centroid_idx].append(idx)
    return clusters


# In[6]:


def closest_centroid(sample, centroids):
    distances = [calc_distance(sample, cen) for cen in centroids]
    closest_idx = np.argmin(distances)
    return closest_idx


# In[7]:


def calc_centroids(X, clusters, k, n_samples, n_features):
    centroids = np.zeros((k, n_features))
    for idx, cluster in enumerate(clusters):
        if cluster == []:
            centroids[idx] = X[np.random.randint(0, n_samples - 1, 1)]
        else:
            centroids[idx] = np.mean(X[cluster], axis=0)
    return centroids


# In[8]:


def calc_cluster_label(clusters, n_samples):
    labels = np.empty(n_samples)
    for c_idx, cluster in enumerate(clusters):
        for s_idx in cluster:
            labels[s_idx] = c_idx
    return labels


# In[9]:


def calc_sse(X, centroids, clusters):
    sse = 0
    for idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            sse += calc_distance(X[sample_idx], centroids[idx]) ** 2
    return sse

# In[10]:


def calc_entropy(prob):
    return -1 * prob * math.log(prob, 2)


# In[97]:


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



# In[12]:


def plot_sse_vs_k(sses, k):
    plt.figure()
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.xticks(np.arange(min(k), max(k) + 1, 1.0))
    plt.plot(k, sses, 'g-')
    plt.show()


# In[13]:


def plot_nmi_vs_k(nmis, k):
    plt.figure()
    plt.xlabel('Number of Clusters')
    plt.ylabel('NMI')
    plt.xticks(np.arange(min(k), max(k) + 1, 1.0))
    plt.plot(k, nmis, 'g-')
    plt.show()


# In[14]:


def main(filename, k_pref):
    dataset = None
    if filename == 'Dermatology':
        dataset = loadDermatologyDataset()
        k_range = 10
    elif filename == 'Vowels':
        dataset = loadVowelDataset()
        k_range = 10
    elif filename == 'Glass':
        dataset = loadGlassDataset()
        k_range = 6
    elif filename == 'Ecoli':
        dataset = loadEcoliDataset()
        k_range = 5
    elif filename =='Yeast':
        dataset = loadYeastDataset()
        k_range = 7
    elif filename == 'Soybean':
        dataset = loadSoybeanDataset()
        k_range = 5
    else:
        print('Please input correct dataset name.')
        return 
    
    data_X = dataset[:, 0: -1]
    data_Y = dataset[:, -1]

    classes = len(np.unique(data_Y))
    
    k_list = []
    if k_pref == 'c':
        k_list.append(classes)
    elif k_pref == 'l':
        k_list = [i for i in range(2, classes + k_range)]
        sses = []
        nmis = []
    else:
        print('Please input correct K preference.')
        return
        
    for k in k_list:
        sse, clusters, centroids = kMeans(data_X, k, 100)
        nmi = calc_nmi(data_X, data_Y, clusters)
        if k_pref == 'l':
            while len(sses) > 0 and sse > sses[-1]:
                sse, clusters, centroids = kMeans(data_X, k, 100)
                nmi = calc_nmi(data_X, data_Y, clusters)
            sses.append(sse)
            nmis.append(nmi)
        else:
            plt.figure()
            plt_name = filename + 'Dataset(K-Means) K=' + str(k)
            plt.title(plt_name)
            plt.scatter(data_X[:, 0], data_X[:, 1], marker='o', c=data_Y)
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='^', c='r')
            plt.show()
        print('No. of Cluster\t\t\tSSE\t\t\tNMI')
        print('\t', k, '\t\t', sse, '\t', nmi)
    
    if k_pref == 'l':
        plot_sse_vs_k(sses, k_list)
        plot_nmi_vs_k(nmis, k_list)


# In[15]:


if __name__ == '__main__':
    dataset = sys.argv[1]
    k_pref = sys.argv[2]
    main(dataset, k_pref)


