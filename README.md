# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries. 

2.Upload and read the dataset. 

3.Check for any null values using the isnull() function. 

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy. 

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KAVIYA D
RegisterNumber: 212223040089 
*/
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
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

## data.head()
![318630456-311a67fb-1265-447c-938d-0c0272211d09](https://github.com/user-attachments/assets/78518cdf-f83b-43ff-aac1-627bc90c493b)

## data.info()

![318630465-0d68fe8f-ca3a-497c-b849-cc795adc9da7](https://github.com/user-attachments/assets/f5411b66-b928-474f-8132-939ffe9fdef9)

## data.isnull().sum()

![318630486-a6d85076-d39f-4a37-9cc7-4faa482709b6](https://github.com/user-attachments/assets/d90f57fc-59eb-4f16-9dae-23e8401f3549)

## data.value.counts()
![image](https://github.com/user-attachments/assets/8844545f-abcd-493e-80dd-3b6b50eae34b)

## x.head()
![318630552-af19bc4f-4230-42f7-9607-63bb491ecb2f](https://github.com/user-attachments/assets/5ee6b2c5-e6c7-43e1-b872-bad8bd1415a4)

## accuracy
![318630596-27177cea-238a-425c-9bf4-2a71622ebe82](https://github.com/user-attachments/assets/dc5fa789-997e-4f00-83fc-0a87112f706d)

## prediction
![318630618-8d2cd76b-d23e-4894-b001-1061c09409fc](https://github.com/user-attachments/assets/ee0485fc-036a-4527-b6ef-c8778afb3d51)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
