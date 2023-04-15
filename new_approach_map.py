#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 18:50:40 2023

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

df = pd.read_csv('Location_Scores.csv')
df = df.dropna()
df = df[df['scores']>=.70]

# Create a folium map centered on the first location
lat = list(df.lat.values)[0]
lon = list(df.long.values)[0]

map = folium.Map(location=[lat, lon], zoom_start=5)

# Define color and labels for each score range
color_dict = {0.6: 'red', 0.4: 'yellow', 0: 'green'}
label_dict = {0.6: 'high', 0.4: 'mid', 0: 'low'}

# Add markers to the map for each location
for index, row in df.iterrows():
    # Determine the color and label based on the score value
    color = None
    label = None
    for threshold, color_value in color_dict.items():
        if row['scores'] >= threshold:
            color = color_value
            label = label_dict[threshold]
            break
    # Add the marker to the map
    marker = folium.Marker(location=[row['lat'], row['long']], popup=f"Score: {row['scores']}", icon=folium.Icon(color=color))
    marker.add_to(map)

# Display the map
#display(map)
map.save('Interactive_map_high.html')

