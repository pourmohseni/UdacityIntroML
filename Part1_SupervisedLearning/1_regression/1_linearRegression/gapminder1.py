import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# Load the data
bmi_data = pd.read_csv('data.csv')

# Make and fit the linear regression model
# Fit the model and Assign it to bmi_life_model
bmi_model = LinearRegression()
X = bmi_data.iloc[:, -1:].values
y = bmi_data.iloc[:, 1].values
bmi_model.fit(X, y)

# Make a prediction using the model
# Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_model.predict([[21.07931]])

print('prediction: ', laos_life_exp)
