# Import statements
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2]

# Create and fit the model to achieve 100% accuracy on the dataset.
model = SVC(kernel='rbf', gamma=30, C=1)
model.fit(X, y)

# Make predictions.
y_pred = model.predict(X)

# Calculate the accuracy.
acc = accuracy_score(y, y_pred)
print('accuracy = ', acc)