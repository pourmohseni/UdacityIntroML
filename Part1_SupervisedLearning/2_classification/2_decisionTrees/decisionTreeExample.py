import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

"""
    This is an example on classification using decision trees.
    The dataset is in surviveData.csv file where the last column are labels.
"""

random.seed(42)

# Load the dataset
in_file = 'surviveData.csv'
full_data = pd.read_csv(in_file)

# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
features_raw = full_data.drop('Survived', axis=1)

# Removing the names
features_no_names = features_raw.drop(['Name'], axis=1)

# One-hot encoding
features = pd.get_dummies(features_no_names)
print("total features: ", len(features.iloc[0]))

# Fill in any blanks with zeroes.
features = features.fillna(0.0)

# Training the model
print('training the model...')


X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2, random_state=42)
print("split done")

# Define the classifier, and fit it to the data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("fitting done")

# ## Testing the model
# Making predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate the accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)

# Improving the model
print('improving the model...')
param_grid = {"max_depth": range(1, 21), "min_samples_leaf": range(1, 10), "min_samples_split": range(2, 10)}
# rain the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("fitting done")

# Make predictions
grid = GridSearchCV(model, param_grid)
grid.fit(X_train, y_train)

# get the best model
print("performing grid search")
best_model = grid.best_estimator_
print("best model params: ", grid.best_params_)

# Making predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Calculate the accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)
