#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 16:50:00 2023

@author: rupasree
"""

import folium
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, auc, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import numpy as np
import xgboost as xgb
from pandas_profiling import ProfileReport


def drop_columns(df, columns):
    df = df.drop(columns, axis=1)
    return df

def create_profile(df, name):
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file(name)

#df = pd.read_csv('Fire_nonFire_Data.csv')
df = pd.read_csv('Simple_dataset2.csv')

columns_to_drop = ['Unnamed: 0', 'ADDRESS', 'fire_count']

df = drop_columns(df, columns_to_drop)

df_one = df[df['Incident']==0]
#create_profile(df_one, 'Fire Instances')

df_dummies = pd.get_dummies(df)

# ensure both train and test set has both(0 and 1) labels
df_0 = df_dummies[df_dummies['Incident']==0]
df_1 = df_dummies[df_dummies['Incident']==1]


X0_train, X0_test, y0_train, y0_test = train_test_split(df_0.drop('Incident', axis=1), df_0['Incident'], test_size=0.33, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(df_1.drop('Incident', axis=1), df_1['Incident'], test_size=0.33, random_state=42)

# merge both labeled instances
X_train = pd.concat([X0_train, X1_train], ignore_index=True)
X_test = pd.concat([X0_test, X1_test], ignore_index=True)
y_train = pd.concat([y0_train, y1_train], ignore_index=True)
y_test = pd.concat([y0_test, y1_test], ignore_index=True)

# concat X_train, y_train to shuffle
train = pd.concat([X_train, y_train], axis=1)

# concat X_test, y_test to shuffle
test = pd.concat([X_test, y_test], axis = 1)

# shuffle train, test
shuffled_train = shuffle(train)
shuffled_test = shuffle(test)

# separate X, y
strain_y = shuffled_train['Incident']
strain_X = shuffled_train.drop('Incident', axis=1)

stest_y = shuffled_test['Incident']
stest_X = shuffled_test.drop('Incident', axis=1)

# concat train test for prediction in cross validation
X_ = pd.concat([strain_X, stest_X], ignore_index=True)
y = pd.concat([strain_y, stest_y], ignore_index=True)

X = drop_columns(X_, ['LAT', "LON"])

# Specify the number of folds
n_folds = 5

# Initialize the stratified k-fold cross-validator
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

scale_pos_weight = sum(strain_y == 0) / sum(strain_y == 1)

# Initialize the XGBoost classifier
xgb_clf = xgb.XGBClassifier(scale_pos_weight = scale_pos_weight)

# Compute the cross-validation scores for the classifier
cv_scores_recall = cross_val_score(xgb_clf, strain_X, strain_y, cv=skf, scoring='recall')
cv_scores_prec = cross_val_score(xgb_clf, strain_X, strain_y, cv=skf, scoring='precision')
cv_scores_rocAuc = cross_val_score(xgb_clf, strain_X, strain_y, cv=skf, scoring='roc_auc')

# Perform cross-validation and get the predicted labels
cv_preds_scores = cross_val_predict(xgb_clf, strain_X, strain_y, cv=skf, method='predict_proba')
y_preds = cv_preds_scores.argmax(axis=1)

print("Number of fires in data: ", np.sum(y == 1))

# Print the cross-validation scores

print("Print the cross-validation scores")
print("Cross-validation scores using recall:", cv_scores_recall)
print("Mean cross-validation score using recall:", np.mean(cv_scores_recall))

print("Cross-validation scores using precision:", cv_scores_prec)
print("Mean cross-validation score using precision:", np.mean(cv_scores_prec))

print("Cross-validation scores using roc_auc:", cv_scores_rocAuc)
print("Mean cross-validation score using roc_auc:", np.mean(cv_scores_rocAuc))

# train on training set
stest_X_save = stest_X.copy()
stest_X = drop_columns(stest_X, ['LON', 'LAT'])

strain_X_save = strain_X.copy()
strain_X = drop_columns(strain_X, ['LON', 'LAT'])

xgb_clf.fit(strain_X, strain_y)

y_pred = xgb_clf.predict(stest_X)
pred_scores = xgb_clf.predict_proba(stest_X)

print("On test data")
cm = confusion_matrix(stest_y, y_pred)

print("Number of fires in test data: ", stest_y.value_counts()[1])

print("Number of false negatives: ", cm[1,0])
print("Number of false positives: ", cm[0,1])
print("Number of true negatives: ", cm[1,1])
print("Number of true positives: ", cm[0,0])


# save the model to a file
with open('xgb_default.pkl', 'wb') as f:
    pickle.dump(xgb_clf, f)

# Get the feature importance scores
importance = xgb_clf.feature_importances_

# Normalize the feature importance scores
importance = importance / np.sum(importance)

'''
# Print the feature importance list
for i, score in enumerate(importance):
    print('Feature %d: %.5f' % (i+1, score))
'''

feat_imp = list(importance)
columns = X.columns

# create a DataFrame
df = pd.DataFrame({'Feature Name': columns})

# create a new DataFrame with the list as a column
new_df = pd.DataFrame({'Feature Importance': feat_imp})

# merge the two DataFrames
merged_df = pd.concat([df, new_df], axis=1)
merged_df.to_csv('Feature Importance List.csv')

# pick the maximum 20 rows based on values in 'Feature Importance'
Top20_Features = merged_df.nlargest(20, 'Feature Importance')

# print the resulting dataframe
print(Top20_Features)

# create interactive map

# Create a dataframe with some sample data
data = {
    'lat': stest_X_save.LAT.values,
    'long': stest_X_save.LON.values,
    'scores': pd.Series(pred_scores[:, 1]),
    'y': stest_y.values,
    'y_pred': list(y_pred)
}

df = pd.DataFrame(data)
df.to_csv('Predicted Scores by Location.csv')
'''
# sort and count
df_sorted= df.sort_values('scores', ascending=False)

df_sorted = df_sorted.head(100)
    
count = ((df_sorted['y'] == 1) & (df_sorted['y_pred']==1)).sum()
#print("Matched predictions out of 20 top-scored instances:", count)
#print("Count", count)
'''