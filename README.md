# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
date: 14/9/23
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries such as pandas module to read the corresponding csv file.
2. Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the corresponding dataset values.
4. Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.
5. Predict the values of array using the variable y_pred.
6. Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.
7. Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.
8. End the program.

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Bala Umesh 
RegisterNumber:  212221040024
``` py
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```


## Output:
### head of the data:
![image](https://github.com/BalaUmesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113031742/91776864-fd6b-4aed-a7aa-8c651f643638)


### copy head of the data:
![image](https://github.com/BalaUmesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113031742/30066172-ec01-4f01-848e-096e5c1a401d)


### null and sum:
![image](https://github.com/BalaUmesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113031742/29bd9cbd-ab37-4a3c-a40c-db5128e10ad2)


### duplicated:
![image](https://github.com/BalaUmesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113031742/c5df2c7f-ca5a-4276-b76e-99ae7f707c9c)


### x-value:
![image](https://github.com/BalaUmesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113031742/111f4890-299d-447e-afbf-a6bb89d1e43f)


### y-value:
![image](https://github.com/BalaUmesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113031742/f1dd052c-4850-40a8-b990-e74c512e4d34)


### accuracy:
![image](https://github.com/BalaUmesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113031742/f87370c2-a055-4b70-849c-4d061e5580c4)


### confusion matrix:
![image](https://github.com/BalaUmesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113031742/decf595a-75ce-4a55-81ae-4a9ab2c8a14f)


### classification report:

![image](https://github.com/BalaUmesh/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/113031742/864a413b-67f7-4b3a-b833-78b045a7e8da)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
