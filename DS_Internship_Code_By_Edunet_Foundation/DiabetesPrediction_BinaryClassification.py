#Daibetes Prediction check whether patient have diabetes or not (1 is yes and 0 is no)
#Binary Classification

# Step 1: Import library
import pandas as pd

# Step 2: Import data
daibetes = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Diabetes.csv')
daibetes.head

# Step 3: Define target(y) and features(x)
daibetes.columns
y = daibetes['daibetes']
x = daibetes[['pregnancies', 'glucose', 'daistolic', 'triceps', 'insuline']]

# Step 4: Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2529)

# Step 5: Select Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Step 6: Tain model (fit model)
model.fit(x_train,y_train)

#model.intercept_
#model.coef_

# Step 7: Prediction
y_pred= model.predict(x_test)

# Step 8: Accuracy
from sklearn.metrics import mean_absolute_percentage_error
mean_absolute_percentage_error(y_test,y_pred)