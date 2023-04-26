# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:19:51 2023

@author: deyr
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Predicted Scores by Location.csv")

df_sorted = df.sort_values('scores', ascending=False)

y = df_sorted['y'].values.tolist

df_20 = df_sorted.head(20)

fire_count = []
y_list = df_20['y'].values.tolist

cnt = 0
for i in range(0, len(y_list)):
    if y_list[i]==1:
        cnt+=1
    
    fire_count.append(cnt)
    
x = [i+1 for i in range(0, len(y_list))]
plt.plot(x, fire_count)