#SARAH SALEEM AKHTER
#Using Polynomial Regression to predict the Housing price according to the ID is assigned to every-house.

#Importing the libraries for the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('housing_price.csv')
ids = dataset.iloc[:, 0:1].values
saleprice = dataset.iloc[:, 1:2].values

#Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regr = LinearRegression()
lin_regr.fit(ids, saleprice)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regr = PolynomialFeatures(degree=5)
ids_poly = poly_regr.fit_transform(ids)
poly_regr.fit = (ids_poly, saleprice)
lin_regr_2 = LinearRegression()
lin_regr_2.fit(ids_poly, saleprice)

#Visualising the Linear Regression results
plt.scatter(ids, saleprice, color= 'purple')
plt.plot(ids, lin_regr.predict(ids), color = 'yellow', label = 'The Best Fit Line')
plt.title('IDs vs saleprice (prediction with linear regression)')
plt.xlabel('IDs')
plt.ylabel('Saleprice')
plt.legend()
plt.show()

#Visualising the Polynomial Regression results
plt.scatter(ids, saleprice, color= 'red')
plt.plot(ids, lin_regr_2.predict(poly_regr.fit_transform(ids)), color = 'blue', label = 'The Best Fit Line')
plt.title('IDs vs saleprice (prediction with polynomial regression)')
plt.xlabel('IDs')
plt.ylabel('Saleprice')
plt.legend()
plt.show()

#Predicting a new result with Linear Regression
hp2920 = lin_regr.predict([[2920]])
hp2925 = lin_regr.predict([[2925]])
hp2930 = lin_regr.predict([[2930]])


#Predicting a new result with Polynomial Regression
hp_poly2920 = lin_regr_2.predict(poly_regr.fit_transform([[2920]]))
hp_poly2925 = lin_regr_2.predict(poly_regr.fit_transform([[2925]]))
hp_poly2930 = lin_regr_2.predict(poly_regr.fit_transform([[2930]]))

print("A/c to Linear Regression: ")
print("The Housing Price for ID number 2920 will be= ", hp2920)
print("The Housing Price for ID number 2925 will be= ", hp2925)
print("The Housing Price for ID number 2930 will be= ", hp2930)

print("A/c to polynomial Regression: ")
print("The Housing Price for ID number 2920 will be= ", hp_poly2920)
print("The Housing Price for ID number 2925 will be= ", hp_poly2925)
print("The Housing Price for ID number 2930 will be= ", hp_poly2930)

