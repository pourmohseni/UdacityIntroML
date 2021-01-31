# Import statements
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

"""
    This is a simple example for classification using decision trees.
    The dataset is in data.csv file where the last column are labels.
"""

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))

# Assign the features and the labels.
X = data[:, 0:2]
y = data[:, 2]

# Create the decision tree model.
model = DecisionTreeClassifier()

# Fit the model.
model.fit(X, y)

# Make predictions.
y_pred = model.predict(X)

# Calculate the accuracy.
acc = accuracy_score(y, y_pred)

print('accuracy = ', acc)
