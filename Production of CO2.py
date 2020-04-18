#SARAH SALEEM AKHTER
#Using Polynomial Regression to predict the temperature of CO2 

#Importing the libraries for the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('global_co2.csv')
year = dataset.iloc[:, 0:1].values
total = dataset.iloc[:, 1:2].values

#Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regr = LinearRegression()
lin_regr.fit(year, total)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regr = PolynomialFeatures(degree=5)
year_poly = poly_regr.fit_transform(year)
poly_regr.fit = (year_poly, total)
lin_regr_2 = LinearRegression()
lin_regr_2.fit(year_poly, total)

#Visualising the Linear Regression results
plt.scatter(year, total, color= 'red')
plt.plot(year, lin_regr.predict(year), color = 'blue', label = 'Best Fit Line')
plt.title('Years vs CO2 Produced (Prediction by Linear Regression)')
plt.xlabel('Years')
plt.ylabel('CO2 Produced')
plt.legend()
plt.show()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

#Visualising the Polynomial Regression results
plt.scatter(year, total, color= 'red')
plt.plot(year, lin_regr_2.predict(poly_regr.fit_transform(year)), color = 'blue', label = 'Best Fit Line')
plt.title('Years vs CO2 Produced (Prediction by Polynomial Regression)')
plt.xlabel('Years')
plt.ylabel('CO2 Produced')
plt.legend()
plt.show()

#Predicting a new result with linear regression
Prod_CO2_11 = lin_regr.predict([[2011]])
Prod_CO2_12 = lin_regr.predict([[2012]])
Prod_CO2_13 = lin_regr.predict([[2013]])

#Predicting a new result with polynomial regression
Prod_CO2_poly11 = lin_regr_2.predict(poly_regr.fit_transform([[2011]]))
Prod_CO2_poly12 = lin_regr_2.predict(poly_regr.fit_transform([[2012]]))
Prod_CO2_poly13 = lin_regr_2.predict(poly_regr.fit_transform([[2013]]))

 
print("Prediction for Production of CO2 in 2011: ")
print("A/c to Linear Regression, CO2 produced in 2011 will be: ", Prod_CO2_11)
print("A/c to polynomial Regression, CO2 produced in 2011 will be: ", Prod_CO2_poly11)

print("Prediction for Production of CO2 in 2012: ")
print("A/c to Linear Regression, CO2 produced in 2012 will be: ", Prod_CO2_12)
print("A/c to polynomial Regression, CO2 produced in 2012 will be: ", Prod_CO2_poly12)

print("Prediction for Production of CO2 in 2013: ")
print("A/c to Linear Regression, CO2 produced in 2013 will be: ", Prod_CO2_13)
print("A/c to polynomial Regression, CO2 produced in 2013 will be: ", Prod_CO2_poly13)