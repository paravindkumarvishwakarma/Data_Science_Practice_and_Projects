"""Chance of Admission for Higher Studies
    Predict the chances of admission of a student to a Graduate program based on:
        1. GRE Scores (20- to 340)
        2. TOEFL Scores (82 to 120)
        3. University Rating (1 to 5)
        4. Statement of Purpose (1 to 5)
        5. Letter of Recommendation Strength (1 to 5)
        6. Undergraduate CGPA (6.8 to 9.92)
        7. Research Experiance (0 to 1)
        8. Chance of Admit (0.34 to 0.97)"""

# Step 1: Import library
import pandas as pd

# Step 2: Import Data
admission = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Admission%20Chance.csv')

admission.head()
admission.info()
admission.describe()

# Step 3: Define target (y) and features (x)
admission.columns
y = admission['Chance of admit']
x = admission.drop(['Serial No','Chance of Admit'],axis=1)

# Step 4: Train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=2529)

#Check shape of train and test sample
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# Step 5: Select model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Step 6: Train or fit model
model.fit(x_train,y_train)
LinearRegression()
model.intercept_
model.coef_

# Step 7: Predict model
y_pred = model.predict(x_test)
y_pred

# Step 8: Model Accuracy
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
mean_absolute_error(y_test,y_pred)
mean_absolute_percentage_error(y_test,y_pred)
