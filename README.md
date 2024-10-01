# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: 

Start the Program

Step 2:

import pandas module and import the required data set.

Step 3:

Find the null values and count them.

Step 4:

Count number of left values.

Step 5:

From sklearn import LabelEncoder to convert string values to numerical values.

Step 6:

From sklearn.model_selection import train_test_split.

Step 7:

Assign the train dataset and test dataset.

Step 8:

From sklearn.tree import DecisionTreeClassifier.

Step 9:

Use criteria as entropy.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:DILLIARASU M 
RegisterNumber:212223230049  
*/

import pandas as pd
data = pd.read_csv("Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
# DATA #

![319277847-fa4bf578-75b3-4a80-88e3-dd2571f963c6](https://github.com/user-attachments/assets/dd06daca-48ba-4afd-9b19-aa85ec81d90b)
# Accuracy:

![319278352-383238e4-b8fc-4a1b-af03-bad654be3103](https://github.com/user-attachments/assets/4d05f743-0c4e-428c-8572-560c92f4d9be)
# Predict:

![319279293-ee3d2dd0-989b-47fd-88ab-12908477c844](https://github.com/user-attachments/assets/7593e2b3-901d-4ed1-9162-bee3f6554548)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
