"""Fish Weight Prediction
With a dataset of fish species, with some of it characteristic like it vertical, diagonal, length, height, and width. We will try to predict the weight of the fish based on their characteristic. We will use Linear Regression Method to see whether the weight of the fish related to their characteristic.

    1. Species: Species name of fish
    2. Weight: Weight of fish in gram
    3. Length1: Vertical length in cm
    4. Length2: Diagonal length in cm
    5. Length3: Cross length in cm
    6. Height: Height in cm
    7. Width: Diagonal width in cm"""

# Step 1 : import library
import pandas as pd

# Step 2 : import data
fish = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Fish.csv')
fish.head()
fish.info()
fish.describe()

# Step 3 : define target (y) and features (X)
fish.columns
y = fish['Weight']
X = fish[['Category','Height', 'Width', 'Length1','Length2', 'Length3']]

# Step 4 : train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

# check shape of train and test sample
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Step 5 : select model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Step 6 : train or fit model
model.fit(X_train,y_train)
model.intercept_
model.coef_

# Step 7 : predict model
y_pred = model.predict(X_test)
y_pred

# Step 8 : model accuracy
from sklearn.metrics import mean_absolute_error, r2_score
mean_absolute_error(y_test,y_pred)
r2_score(y_test,y_pred)
