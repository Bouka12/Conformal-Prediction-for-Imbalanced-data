#!/usr/bin/env python

"""
Example: inductive conformal classification using DecisionTreeClassifier
"""

# Authors: Henrik Linusson

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from nonconformist.base import ClassifierAdapter
from nonconformist.cp import TcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc
from nonconformist.evaluation import class_mean_errors

# -----------------------------------------------------------------------------
# Setup training, calibration and test indices
# -----------------------------------------------------------------------------
data = pd.read_csv(r'C:\Users\BOUKA\OneDrive\Bureau\MUICE\AP\Proyecto\nonconformist\examples\diabetes.csv')
data.rename(columns={'Outcome':'target'}, inplace=True)

# Assuming 'X' contains the features and 'y' contains the target variable
X = data.drop(columns=['target'])  # Adjust the target_column_name
y = data['target']  # Adjust the target_column_name

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------------------------------------------------------
# Train and calibrate TCP
# -----------------------------------------------------------------------------
tcp = TcpClassifier(
	ClassifierNc(
		ClassifierAdapter(RandomForestClassifier(random_state=42)),
		MarginErrFunc()
	)
)

tcp.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
X_test = pd.DataFrame(X_test)

prediction = tcp.predict(X_test.values, significance=0.1)
header = np.array(['c0','c1', 'Truth'])
table = np.vstack([prediction.T, y_test]).T
df = pd.DataFrame(np.vstack([header, table]))
print('TCP')
print('---')
print(df)

error_rate = class_mean_errors(tcp.predict(X_test.values), y_test, significance=0.1)
print('Error rate: {}'.format(error_rate))





# -----------------------------------------------------------------------------
# Train and calibrate Mondrian (class-conditional) TCP
# -----------------------------------------------------------------------------
tcp = TcpClassifier(
	ClassifierNc(
		ClassifierAdapter(RandomForestClassifier(random_state=42)),
		MarginErrFunc()
	),
	condition=lambda x: x[1],
)

tcp.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
X_test = pd.DataFrame(X_test)

prediction = tcp.predict(X_test.values, significance=0.1)
header = np.array(['c0','c1','Truth'])
table = np.vstack([prediction.T, y_test]).T
df = pd.DataFrame(np.vstack([header, table]))
print('\nClass-conditional TCP')
print('---------------------')
print(df)

error_rate = class_mean_errors(tcp.predict(X_test.values), y_test, significance=0.1)
print('Error rate: {}'.format(error_rate))


# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
rf_predictions = rf_classifier.predict(X_test)

# Calculate the error rate
rf_error_rate = 1 - np.mean(rf_predictions == y_test)
print('Baseline Random Forest Error rate: {:.2f}'.format(rf_error_rate))