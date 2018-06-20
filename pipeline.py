# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:14:40 2018

@author: Abhishek

Streamlining workﬂows with pipelines
------------------------------------
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
#reading in the dataset directly from the UCI website using pandas
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data',
                 header=None)
# Using a LabelEncoder object, we transform the class labels from their 
#original string representation ('M' and 'B') into integers
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
#divide the dataset into a separate training dataset (80 percent of the data)
#and a separate test dataset (20 percent of the data
X_train, X_test, y_train, y_test = \
train_test_split(X, y,test_size=0.20,stratify=y,random_state=1)

#The make_pipeline function takes an arbitrary number of scikit-learn transformer
pipe_lr = make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))