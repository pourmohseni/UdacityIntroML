# Import statements
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

# Import the train test split
from sklearn.model_selection import train_test_split

# Read in the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:, 0:2]
y = data[:, 2]

# Use train test split to split your data with test size of 25% and random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Instantiate your decision tree model
model = DecisionTreeClassifier()

# optional grid search
# grid = {'max_depth': np.arange(2, 20), 'min_samples_split': np.arange(2,20), 'min_samples_leaf': np.arange(1,20)}
# model = GridSearchCV(DecisionTreeClassifier(), grid)

# Fit the model.
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy.
acc = accuracy_score(y_test, y_pred)

print('accuracy =', acc)
