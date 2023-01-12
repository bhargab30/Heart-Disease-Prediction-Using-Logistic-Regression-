import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv('url for the heart data downloaded from kaggle')

#checking for missing values
heart_data.isnull().sum()

#statistical measures about the data
heart_data.describe()

# if dropping column axis = 1 if dropping row axis = 0
#dropping means it will remove that row or column
X = heart_data.drop(columns ='target',axis = 1)
Y = heart_data['target']

print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, stratify = Y, random_state=2)

model = LogisticRegression()
#training the LogisticRegression model with training data
model.fit(X_train,Y_train)

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data: ', training_data_accuracy)
print('Accuracy on test data: ', test_data_accuracy)
