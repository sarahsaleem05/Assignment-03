#SARAH SALEEM AKHTER
#Using Polynomial Regression to predict the Data of monthly experience and income distribution of different employees

# Importing the libraries and the dataset for evaluation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('monthlyexp vs incom.csv')
months_exp = dataset.iloc[:, 0:1].values
income = dataset.iloc[:, 1:2].values

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regr = LinearRegression()
lin_regr.fit(months_exp, income)

## Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regr = PolynomialFeatures(degree = 2)
months_exp_poly = poly_regr.fit_transform(months_exp)
poly_regr.fit(months_exp_poly, income)
lin_regr_2 = LinearRegression()
lin_regr_2.fit(months_exp_poly, income)

#Visualising the Linear Regression results
plt.scatter(months_exp, income, color= 'purple')
plt.plot(months_exp, lin_regr.predict(months_exp), color = 'blue', label = 'The Best Fit Line')
plt.title('months experience vs income ( by linear regression)')
plt.xlabel('Months Experience')
plt.ylabel('Income')
plt.legend()
plt.show()

#Visualising the Polynomial Regression results
plt.scatter(months_exp, income, color= 'green')
plt.plot(months_exp, lin_regr_2.predict(poly_regr.fit_transform(months_exp)), color = 'blue', label = 'The Best Fit Line')
plt.title('months experience vs income (by polynomial regression)')
plt.xlabel('Months Experience')
plt.ylabel('Income')
plt.legend()
plt.show()

#Predicting a new result with Linear Regression
exp_19 = lin_regr.predict([[19]])
exp_20 = lin_regr.predict([[20]])
exp_21 = lin_regr.predict([[21]])

#Predicting a new result with Polynomial Regression
exp_poly19 = lin_regr_2.predict(poly_regr.fit_transform([[19]]))
exp_poly20 = lin_regr_2.predict(poly_regr.fit_transform([[20]]))
exp_poly21 = lin_regr_2.predict(poly_regr.fit_transform([[21]]))

print("By Linear Regression: ")
print("Income= ", exp_19)
print("Income= ", exp_20)
print("Income= ", exp_21)

print("By Polynomial Regression: ")
print("Income= ", exp_poly19)
print("Income= ", exp_poly20)
print("Income= ", exp_poly21)