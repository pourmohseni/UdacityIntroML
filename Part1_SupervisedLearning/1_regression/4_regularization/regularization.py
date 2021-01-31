# Add import statements
import pandas as pd
from numpy import all
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Assign the data to predictor and outcome variables
# Load the data
train_data = pd.read_csv('data.csv', header=None)
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

# Create the standardization scaling object.
minMax_scaler = MinMaxScaler()
scaler = StandardScaler()

# Fit the standardization parameters and scale the data.
X_scaled_standard = scaler.fit_transform(X)
X_scaled_minMax = minMax_scaler.fit_transform(X)

# Create the linear regression model with lasso regularization.
lasso_reg_plain = Lasso(alpha=1.0)
lasso_reg_minMax = Lasso(alpha=1.0)
lasso_reg_stand = Lasso(alpha=1.0)

# Fit the model.
lasso_reg_plain.fit(X, y)
lasso_reg_minMax.fit(X_scaled_minMax, y)
lasso_reg_stand.fit(X_scaled_standard, y)

# Retrieve and print out the coefficients from the regression model.
reg_coef_plain = lasso_reg_plain.coef_
reg_minMax_coef = lasso_reg_minMax.coef_
reg_minMax_stand = lasso_reg_stand.coef_

reg_coef = reg_coef_plain
print('coefficients: ', reg_coef)