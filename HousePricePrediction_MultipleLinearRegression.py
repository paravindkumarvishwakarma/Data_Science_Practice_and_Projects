""" Ordinary Least Squares (OLS) is a method used in linear regression analysis to estimate the unknown parameters in a linear regression model. The goal of OLS is to minimize the sum of the squares of the differences between the observed and predicted values of the dependent variable."""

#House Price Prediction

# Step 1: Import library
import pandas as pd #pip install pandas

# Step 2: Import data
house = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/House%20Prices.csv')
house.head

# Step 3: Define target(y) and features(x)
house.columns
y = house['MEDV']
x = house[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NX', 'RM', 'AGE', 'DIS', ]]

# Step 4: Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2529)

# Step 5: Select Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Step 6: Tain model (fit model)
model.fit(x_train,y_train)

#model.intercept_
#model.coef_

# Step 7: Prediction
y_pred= model.predict(x_test)
x_test

# Step 8: Accuracy
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)