"""Customer Purchase Prediction & Effect of Micro-Numerosity"""

# Step 1 : import library
import pandas as pd

# Step 2 : import data
purchase = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Customer%20Purchase.csv')
purchase.head()
purchase.info()
purchase.describe()

# Step 3 : define target (y) and features (X)
purchase.columns
y = purchase['Purchased']
X = purchase.drop(['Purchased','Customer ID'],axis=1)

# encoding categorical variable
X.replace({'Review':{'Poor':0,'Average':1,'Good':2}},inplace=True)
X.replace({'Education':{'School':0,'UG':1,'PG':2}},inplace=True)
X.replace({'Gender':{'Male': 0,'Female':1}},inplace=True)
          
# display first 5 rows
X.head()

# Step 4 : train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=2529)

# check shape of train and test sample
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Step 5 : select model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# Step 6 : train or fit model
model.fit(X_train,y_train)

# Step 7 : predict model
y_pred = model.predict(X_test)
y_pred

# Step 8 : model accuracy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))