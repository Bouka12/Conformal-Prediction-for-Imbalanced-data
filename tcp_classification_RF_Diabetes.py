#!/usr/bin/env python

"""
Transductive conformal classification using Random Forest Classifier for Pima Indian Diabetes Classification
"""

# Examples provided in the `nonconformist` python package for conformal prediction

# Modified by Mabrouka Salmi

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from nonconformist.base import ClassifierAdapter
from nonconformist.cp import TcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc
from nonconformist.evaluation import class_mean_errors,  class_avg_c, class_n_correct

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

# Validity
error_rate = class_mean_errors(tcp.predict(X_test.values), y_test, significance=0.1)
print('Error rate:- Transductive CP {}'.format(error_rate))

# Efficiency
avg_c_class = class_avg_c(tcp.predict(X_test.values), y_test, significance=0.1)
"""Calculates the average number of classes per prediction of a conformal classification model."""
print('Average number of classes per prediction- Transductive CP:{}'.format(avg_c_class))

# Accuracy
n_correct_class = class_n_correct(tcp.predict(X_test.values), y_test, significance=0.1)
"""Calculates the number of correct predictions made by a conformal classification model."""
print('Accuracy of the Transductive CP: {}'.format(n_correct_class/X_test.shape[0]))


'''
Error rate:- Transductive CP 0.11688311688311692
Average number of classes per prediction- Transductive CP:1.3311688311688312
Accuracy of the Transductive CP: 0.8766233766233766
'''

# -----------------------------------------------------------------------------
# Train and calibrate Mondrian (class-conditional) TCP
# -----------------------------------------------------------------------------
tcp1 = TcpClassifier(
	ClassifierNc(
		ClassifierAdapter(RandomForestClassifier(random_state=42)),
		MarginErrFunc()
	),
	condition=lambda x: x[-1],
)

tcp1.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------------
X_test = pd.DataFrame(X_test)

prediction = tcp1.predict(X_test.values, significance=0.1)
header = np.array(['c0','c1','Truth'])
table = np.vstack([prediction.T, y_test]).T
df = pd.DataFrame(np.vstack([header, table]))
print('\nClass-conditional Label Conditional TCP')
print('---------------------')
print(df)

error_rate = class_mean_errors(tcp1.predict(X_test.values), y_test, significance=0.1)
print('Error rate - Label-Conditional CP: {}'.format(error_rate))

# Efficiency
avg_c_class = class_avg_c(tcp1.predict(X_test.values), y_test, significance=0.1)
"""Calculates the average number of classes per prediction of a conformal classification model."""
print('Average number of classes per prediction- Label-Conditional CP:{}'.format(avg_c_class))

# Accuracy
n_correct_class = class_n_correct(tcp1.predict(X_test.values), y_test, significance=0.1)
"""Calculates the number of correct predictions made by a conformal classification model."""
print('Accuracy of the Label-Conditional CP: {}'.format(n_correct_class/X_test.shape[0]))



'''
Error rate - Label-Conditional CP: 0.11688311688311692
Average number of classes per prediction- Label-Conditional CP:1.3831168831168832
Accuracy of the Label-Conditional CP: 0.8831168831168831
'''


# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
rf_predictions = rf_classifier.predict(X_test)

# Calculate the error rate
rf_error_rate = 1 - np.mean(rf_predictions == y_test)
print('Baseline Random Forest Error rate: {:.2f}'.format(rf_error_rate))


# Accuracy
acc_RF = accuracy_score(rf_predictions, y_test)
"""Calculates the number of correct predictions made by a conformal classification model."""
print('Accuracy of the Random Forest: {}'.format(acc_RF))


# Perform cross-validation to obtain accuracy scores
cv_accuracy_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)

# Calculate the mean and standard deviation of cross-validation accuracy
cv_accuracy_mean = cv_accuracy_scores.mean()

# Print the results
print("Random Forest Cross-Validation Accuracy Mean:", cv_accuracy_mean)
'''
Baseline Random Forest Error rate: 0.28
Accuracy of the Random Forest: 0.7207792207792207
'''
