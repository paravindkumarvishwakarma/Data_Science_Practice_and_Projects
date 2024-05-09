# Step 1: Import libraries
import pandas as pd

# Step 2: Import Data
salary = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Salary%20Data.csv')

salary.head()
salary.info()
salary.describe()

# Step 3: Define target (y) and features (X)
""" Experiance predict with the help of salary
    predict salary with the help of experiance
                     Y                  X
            (A)     Exp    <--------- Salary 
            (B)     Salary <--------- Exp 

    This is a supervise machine learning(Regression) -> Salary is a continuous variable
"""
salary.columns

y = salary['Salary'] # y is dependent variable

x = salary[['Experiance Years']] # x is independent variable

salary.shape

x.shape

y.shape

# Step 4: train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

