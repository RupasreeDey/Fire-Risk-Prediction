#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 01:00:32 2023

@author: rupasree
"""

import folium
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, auc, accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
import xgboost as xgb
from IPython.display import display
import matplotlib as plt

df_init = pd.read_csv('Location_Scores_v2.csv')
df = df_init.dropna()

df = df[df['lat'] >= -90]

import geohash

# Generate geohashes for each latitude and longitude pair
df['geohash'] = df.apply(lambda row: geohash.encode(row['lat'], row['long'], precision=5), axis=1)

from sklearn.cluster import KMeans

# Fit k-means clustering model with latitude and longitude features
kmeans = KMeans(n_clusters=10)
kmeans.fit(df[['lat', 'long']])

# Add cluster labels to DataFrame
df['cluster'] = kmeans.labels_
