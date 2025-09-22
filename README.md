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
Developed by: PRIYANKA S
RegisterNumber: 212224040255 
*/

import pandas as pd
data=pd.read_csv('Employee.csv')
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("NAME:PRIYANKA S")
print("REG NO:212224040255")
print(y_pred)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
classification=classification_report(y_test,y_pred)
print("NAME:PRIYANKA S")
print("REG NO:212224040255")
print("accuracy",accuracy)
print("confusion",confusion)
print("classification",classification)
dt.predict([[10,9,9,66,8,90,90]])

```

## Output:


<img width="1648" height="287" alt="image" src="https://github.com/user-attachments/assets/a76f52ac-87da-4a85-b3bb-f270e446fee3" />


<img width="1641" height="445" alt="image" src="https://github.com/user-attachments/assets/15031131-f1b3-4eec-a15e-16aa799bbf97" />


<img width="1619" height="305" alt="image" src="https://github.com/user-attachments/assets/a37bab43-01b6-4cc0-a698-eeba3e76c07d" />


<img width="1630" height="122" alt="image" src="https://github.com/user-attachments/assets/85264474-b52e-4df4-9fca-dd76968a8ecc" />


<img width="1628" height="292" alt="image" src="https://github.com/user-attachments/assets/8dd519a3-b548-4459-8abd-16427c8aac00" />


<img width="1641" height="304" alt="image" src="https://github.com/user-attachments/assets/8d0bc25d-c86e-4a3d-8ff7-b49e68d12737" />


<img width="1650" height="122" alt="image" src="https://github.com/user-attachments/assets/6ab85cee-a89f-4160-b644-80ee885ae996" />


<img width="1653" height="86" alt="image" src="https://github.com/user-attachments/assets/4f7c8ccc-28b2-4102-9a72-ccab72df95e9" />


<img width="1651" height="353" alt="image" src="https://github.com/user-attachments/assets/717e7ed4-c938-4b61-8cc3-615c9b09e388" />


<img width="1654" height="136" alt="image" src="https://github.com/user-attachments/assets/b5e4ecfc-580b-4710-a6ad-10742a793174" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
