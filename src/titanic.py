# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:20:18 2020

@author: v01d13
"""

import numpy as np
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
X_train = dataset.iloc[:, [2, 4, 5]].values
y_train = dataset.iloc[: , 1].values
X_test = dataset_test.iloc[:, [1, 3, 4]].values

#Fixing the missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X_train[:, 2:3])
imputer_test = imputer.fit(X_test[:, 2:3])
X_train[:, 2:3] = imputer.transform(X_train[:, 2:3])
X_test[:, 2:3] = imputer_test.transform(X_test[:, 2:3])

#Encoding the cateogrical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 0] = labelencoder_X.fit_transform(X_train[:, 0])
X_train[:, 1] = labelencoder_X.fit_transform(X_train[:, 1])
X_test[:, 0] = labelencoder_X.fit_transform(X_test[:, 0])
X_test[:, 1] = labelencoder_X.fit_transform(X_test[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [0, 1])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()
print(X_train)
print(X_test)

#Fitting the classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)