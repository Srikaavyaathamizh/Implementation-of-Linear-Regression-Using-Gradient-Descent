# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import necessary libraries such as NumPy, Pandas, Matplotlib, and metrics from sklearn.
2.Load the dataset into a Pandas DataFrame and preview it using head() and tail().
3.Extract the independent variable X and dependent variable Y from the dataset.
4.Initialize the slope m and intercept c to zero. Set the learning rate L and define the number of epochs.
5.In a loop over the number of epochs:
.Compute the predicted value Y_pred using the formula . Calculate the gradients
.Update the parameters m and c using the gradients and learning rate.
.Track and store the error in each epoch.
6.Plot the error against the number of epochs to visualize the convergence.
7.Display the final values of m and c, and the error plot.
```
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SRIKAAVYAA T
RegisterNumber:  212223230214
*/

import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
![Screenshot 2024-09-03 141853](https://github.com/user-attachments/assets/28aceb33-19d6-447d-93f0-2b0af85e136e)

```
dataset.info()
```

![Screenshot 2024-09-03 142054](https://github.com/user-attachments/assets/c7885606-a022-4ede-a716-7305ff3fc6f9)
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
![Screenshot 2024-09-03 142155](https://github.com/user-attachments/assets/ec8fff5b-845e-4df7-9628-b62c75e52448)

```
print(X.shape)
print(Y.shape)
```
![image](https://github.com/user-attachments/assets/d08a4303-14a2-45cd-887c-f34cedec63af)
```
m=0
c=0
L=0.0001
epochs=5000
n=float(len(X))
error=[]
for i in range(epochs):
    Y_pred = m*X +c
    D_m=(-2/n)*sum(X *(Y-Y_pred))
    D_c=(-2/n)*sum(Y -Y_pred)
    m=m-L*D_m
    c=c-L*D_c
    error.append(sum(Y-Y_pred)**2)
print(m,c)
type(error)
print(len(error))
```
![Screenshot 2024-09-03 142331](https://github.com/user-attachments/assets/20fc0dc0-afe2-4ee4-9830-ac0f39712fd0)

```
plt.plot(range(0,epochs),error)
```
![Screenshot 2024-09-03 142456](https://github.com/user-attachments/assets/4036e4b0-ef43-4ba7-b250-7ac2d0ab70c4)
![Screenshot 2024-09-03 142512](https://github.com/user-attachments/assets/339c3f48-2312-4631-ab7f-1091bd5b63e5)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
