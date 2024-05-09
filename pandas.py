import pandas as pd

titanic = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Titanic.csv') #read data from website
print(titanic)

titanic.head() #show first five lines
titanic.info() #giving the information of data (class - dataframe), no of rows, no of columns, datatypes, (string is known as object), 
titanic.describe()
