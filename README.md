# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the data
2. Split the data into training and testing dataset
3. Train the Linear Regression model
4. Evaluate the model's performance

## Program:

## Developed by: Yamuna M
## RegisterNumber:212223230248

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
data = pd.read_csv('car_price_prediction_.csv')
print(data.head())
X = data[['Mileage', 'Engine Size']] 
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/9ff11213-1920-4a2c-9e93-ee3b31354f7b)

![image](https://github.com/user-attachments/assets/2fd6e717-ed41-4bb2-b66b-fc462ff24a5d)


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
