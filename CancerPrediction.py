"""------------------CANCER PREDICTION------------------
    Dataset information:
    Target variable (y):
        -> Diagnosis (M = maligant, B = benign)
    
    Ten features (x) are computed for each cell nucleus:
        1. radius (mean of distance from center to points on the perimeter)
        2. texture ( standard deviation of gray-scale values)
        3. Perimeter
        4. area
        5. smoothness (local variation in radius lengths)
        6. compactness (perimeter^2/area - 1.0)
        7. concavity (severity of concave portions of the contour)
        8. concave points (number of concave portions of the contour)
        9. symmetry
        10. fractal dimension (coastline approximation - 1)
        
        for each characteristics three measures are given:
            a. Mean
            b. Standard error
            c. Largest/worst"""

# Step 1: Import library
import pandas as pd

# Step 2: Import Data
cancer = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Cancer.csv')
cancer.head()
cancer.info()
cancer.describe()

# Step 3: Define target (y) and features (x)
cancer.columns
y = cancer['diagnosis']
x = cancer.drop(['id','diagnosis','unnamed: 32'],axis=1)

# Step 4: Train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=2529)

#Check shape of train and test sample
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# Step 5: Select model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)

# Step 6: Train or fit model
model.fit(x_train,y_train)
LogisticRegression(max_iter=5000)
model.intercept_
model.coef_

# Step 7: Predict model
y_pred = model.predict(x_test)
y_pred

# Step 8: Model Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
confusion_matrix(y_test, y_pred)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))


