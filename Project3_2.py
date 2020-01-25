# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:29:09 2019

@author: kiran Gupta  ID:1001726759
FOr n-Cluster = 40
"""
import pandas as pd
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

f = open("ATNTFaceImages400.txt", "r")

line = f.readline()
arr = np.array([line.split(",")])
line = f.readline()

while line:    
    newarr = np.array(line.split(","))
    arr = np.append(arr,[newarr],axis=0)
    line = f.readline()

f.close()

y = arr[0,:]
arr = np.array(np.delete(arr,0,0))  
arr = np.array(np.transpose(arr))

arr = arr.astype(np.int)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=40, random_state=0)
kmeans.fit(arr)
labels = kmeans.predict(arr)

k_labels_matched = np.empty_like(kmeans.labels_)
for k in np.unique(kmeans.labels_):
    match_nums = [np.sum((kmeans.labels_==k)*(y==t)) for t in np.unique(y)]
    k_labels_matched[kmeans.labels_==k] = np.unique(y)[np.argmax(match_nums)]
    
'''  
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y.astype(np.int), k_labels_matched)
'''

y_pred = [i+1 for i in kmeans.labels_]
y_pred = np.array(y_pred)

from sklearn.metrics import confusion_matrix
cm1 = pd.DataFrame(confusion_matrix(y.astype(np.int),y_pred))

#Bipartite Graph Reordering
from scipy.optimize import linear_sum_assignment
ind = linear_sum_assignment(np.max(cm1)-cm1)
order = ind[1]
final_cm = cm1.iloc[:,list(order)]  #This is a final Confusion Matrix obtained by Bipartite Graph Matching


#Accuracy of K-Means
from sklearn.metrics import accuracy_score
ac = accuracy_score(y.astype(np.int), k_labels_matched)  #77% Accuracy

'''
from sklearn.metrics import recall_score
recall_score(y.astype(np.int), k_labels_matched, average='macro') 
'''