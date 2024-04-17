#!/usr/bin/env python

"""
Inductive conformal classification using Random Forest Classifier for Pima Indian Diabetes Classification
"""

# Examples provided in the `nonconformist` python package for conformal prediction

# Modified by Mabrouka Salmi



import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from nonconformist.base import ClassifierAdapter
from nonconformist.icp import IcpClassifier
from nonconformist.evaluation import class_mean_errors, class_avg_c, class_n_correct

from nonconformist.nc import ClassifierNc, MarginErrFunc

# -----------------------------------------------------------------------------
# Setup training, calibration and test indices
# -----------------------------------------------------------------------------
''' Work on Pima diabetes dataset: moderately imbalanced and relatively small dataset'''
data = pd.read_csv(r'C:\Users\BOUKA\OneDrive\Bureau\MUICE\AP\Proyecto\nonconformist\examples\diabetes.csv')
data.rename(columns={'Outcome':'target'}, inplace=True)

# Assuming 'X' contains the features and 'y' contains the target variable
X = data.drop(columns=['target'])  # Adjust the target_column_name
y = data['target']  # Adjust the target_column_name

# Split the dataset into training and test sets
X_train_s, X_test, y_train_s, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_train_s, y_train_s, test_size=0.15/0.85, random_state=42)


rte = X_test.shape[0]/X.shape[0]
rtn = X_train.shape[0]/X.shape[0]
rcl = X_calib.shape[0]/X.shape[0]

print("ratio of the train data = ",rtn)
print("ratio of the calib data = ",rcl)
print("ratio of the test data",rte )

# -----------------------------------------------------------------------------
# Train and calibrate
# -----------------------------------------------------------------------------
icp = IcpClassifier(ClassifierNc(ClassifierAdapter(RandomForestClassifier(random_state=42)),
                                 MarginErrFunc()))
icp.fit(X_train.values, y_train)
icp.calibrate(X_calib.values, y_calib)

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
prediction = icp.predict(X_test.values, significance=0.1)
header = np.array(['c0','c1','Truth'])
table = np.vstack([prediction.T, y_test]).T
df = pd.DataFrame(np.vstack([header, table]))
print(df)

# ----------------------------------------------------------------------------
# Evaluate predictions
# ----------------------------------------------------------------------------
# Validity
error_rate = class_mean_errors(icp.predict(X_test.values), y_test, significance=0.1)
print('Error rate - Inductive CP: {}'.format(error_rate))

# Efficiency
avg_c_class = class_avg_c(icp.predict(X_test.values), y_test, significance=0.1)
"""Calculates the average number of classes per prediction of a conformal classification model."""
print('Average number of classes per prediction-Inductive CP:{}'.format(avg_c_class))

# Accuracy
n_correct_class = class_n_correct(icp.predict(X_test.values), y_test, significance=0.1)
"""Calculates the number of correct predictions made by a conformal classification model."""
print('Accuracy of the inductive CP: {}'.format(n_correct_class/X_test.shape[0]))

'''
Error rate: 0.10344827586206895
Average number of classes per prediction:1.3017241379310345
Accuracy of the inductive CP: 0.896551724137931
'''