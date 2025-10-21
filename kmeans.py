# kmeans clustering and assigning sample weight based on cluster information
import ipdb
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import random
import torch
import time
from tqdm import tqdm
from sklearn.manifold import TSNE


import numpy as np

def initialize_centroids(X, k):
    """随机初始化质心"""
    n_samples = X.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    return centroids

def assign_clusters(X, centroids):
    """将每个样本分配到最近的质心"""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def compute_centroids(X, labels, k):
    """根据当前的样本分配更新质心"""
    centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return centroids

def calculate_cluster_proportions(labels, k):
    """计算每个簇的样本比例"""
    total_samples = len(labels)
    proportions = [(labels == i).sum() / total_samples for i in range(k)]
    return proportions


def Kmeans(X, k, max_iters=100, tol=1e-4):
    """实现 k-means 算法"""
    centroids = initialize_centroids(X, k)
    
    for _ in range(max_iters):
        # 分配样本到最近的质心
        labels = assign_clusters(X, centroids)
        
        # 计算新的质心
        new_centroids = compute_centroids(X, labels, k)
        
        # 检查质心是否收敛
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids
    
    proportions = calculate_cluster_proportions(labels, k)
    
    return proportions, centroids, labels


    
for net_id in range(7):

    contextual_prototypes = []
    contextual_priori = []

    feats_i = torch.load('./contextual_feats/contextual_feats_client'+str(net_id)+'.pkl').cpu().numpy() # 加载背景特征

    n_clusters = 32
    contextual_priori, contextual_prototypes, predicts = Kmeans(feats_i, n_clusters, max_iters=30)

    np.save('contextual_prototypes_client'+str(net_id)+'.npy', contextual_prototypes)
    np.save('contextual_priori_client'+str(net_id)+'.npy', contextual_priori)



























32










