# TODO: Add import statements
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib import pyplot as plt

# Load the data
train_data = pd.read_csv('data.csv')
X = train_data[['Var_X']]
y = train_data['Var_Y']

# Create polynomial features
poly_feat = PolynomialFeatures(degree=4)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

X_sort = X.sort_values(by=['Var_X'])
plt.scatter(X, y, color='blue')
plt.plot(X_sort, poly_model.predict(poly_feat.fit_transform(X_sort)), color='red')
plt.show()
