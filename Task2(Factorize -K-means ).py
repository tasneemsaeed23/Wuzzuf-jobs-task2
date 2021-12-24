# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 20:28:32 2021

@author: Tasneem said
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wuzzuf_Jobs.csv')

# fatorize 
dataset['fact'] = pd.factorize(dataset['YearsExp'])[0]

dataset['Title']=pd.factorize(dataset['Title'])[0]
dataset['Company']=pd.factorize(dataset['Company'])[0]

X = dataset.iloc[:, [0, 1]].values
x=pd.DataFrame(X)

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()