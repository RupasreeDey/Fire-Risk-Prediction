#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 11:17:01 2023

@author: rupasree
"""

import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils import shuffle
from pandas_profiling import ProfileReport
from math import radians, cos, sin, asin, sqrt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

def drop_duplicate(df):
    df = df.drop_duplicates()
    return df

def drop_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df

def create_profile(df, name):
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file(name)
    
def single_pt_haversine(lat, lng, degrees=True):
    """
    'Single-point' Haversine: Calculates the great circle distance
    between a point on Earth and the (0, 0) lat-long coordinate
    """
    r = 3956.0 # Earth's radius (km). Have r = 3956 if you want miles

    # Convert decimal degrees to radians
    if degrees:
        lat, lng = map(radians, [lat, lng])

    # 'Single-point' Haversine formula
    a = sin(lat/2.0)**2 + cos(lat) * sin(lng/2.0)**2
    d = 2.0 * r * asin(sqrt(a)) 

    return round(d,7)

df_fict = pd.read_csv('Dataset_fictitious.csv')

# convert lat-lon to single-point feature using haversine
#df_fict['Location'] = [single_pt_haversine(x, y) for x, y in zip(df_fict.LAT, df_fict.LON)]

# drop address and some others from df_fict
columns_to_drop = ['Shape_Length', 'Shape_Area', 'BuildVal','Unnamed: 0', 'ADDRESS', 'Permit Duration']

df_fict_new = drop_columns(df_fict, columns_to_drop)

# ensure both train and test set has both(0 and 1) labels
df_0 = df_fict_new[df_fict_new['Incident']==0]
df_1 = df_fict_new[df_fict_new['Incident']==1]

X0_train, X0_test, y0_train, y0_test = train_test_split(df_0.drop('Incident', axis=1), df_0['Incident'], test_size=0.33, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(df_1.drop('Incident', axis=1), df_1['Incident'], test_size=0.33, random_state=42)


# merge both labeled instances
X_train = pd.concat([X0_train, X1_train], ignore_index=True)
X_test = pd.concat([X0_test, X1_test], ignore_index=True)
y_train = pd.concat([y0_train, y1_train], ignore_index=True)
y_test = pd.concat([y0_test, y1_test], ignore_index=True)

# concat X_train, y_train to shuffle
train = X_train + y_train

# concat X_test, y_test to shuffle
test = X_test + y_test

# shuffle train, test
shuffled_train = shuffle(train)
shuffled_test = shuffle(test)

# separate X, y
strain_y = shuffled_train['Incident']
strain_X = shuffled_train.drop('Incident', axis=1)

stest_y = shuffled_test['Incident']
stest_X = shuffled_test.drop('Incident', axis=1)

# One-hot encode categorical variables
train_data = pd.get_dummies(strain_X)
test_data = pd.get_dummies(stest_X)

# train
xgb_model = xgb.XGBClassifier()

cv_results = xgb.cv(
    params=xgb_model.get_params(),
    dtrain=xgb.DMatrix(train_data, label=strain_y),
    num_boost_round=xgb_model.get_params()['n_estimators'],
    nfold=5,
    metrics='merror',
    early_stopping_rounds=10,
    stratified=True,
    seed=42
)

# Train XGBoost model with optimal number of estimators
xgb_model.set_params(n_estimators=cv_results.shape[0])
eval_set = [(train_data, strain_y), (test_data, stest_y)]
xgb_model.fit(train_data, strain_y, eval_set=eval_set, eval_metric='merror', early_stopping_rounds=10)

# Make predictions on test set
y_pred = xgb_model.predict(test_data)

# Evaluate model performance
accuracy = accuracy_score(stest_y, y_pred)
confusion_matrix = confusion_matrix(stest_y, y_pred)

# Print evaluation results per estimator
evals_result = xgb_model.evals_result()
print("Evaluation results per estimator:")
for i, evals in enumerate(evals_result['validation_0']['merror']):
    print("Estimator {}: training error {:.2f}%, validation error {:.2f}%".format(i+1, evals*100, evals_result['validation_1']['merror'][i]*100))

print("Optimal number of estimators: {}".format(cv_results.shape[0]))
print("Accuracy: {:.2f}%".format(accuracy*100))
print("Confusion Matrix:\n", confusion_matrix)