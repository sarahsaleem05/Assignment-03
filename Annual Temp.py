#SARAH SALEEM AKHTER
# Using Polynomial Regression to predict the temperature in 2016 and 2017 using the past data of both industries

#================================ FOR GCAG===================================
# Importing the libraries and the dataset for evaluation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('ann_temp_gcag.csv')
year = dataset.iloc[:, 1:2].values
mean = dataset.iloc[:, 2:3].values

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regr = LinearRegression()
lin_regr.fit(year, mean)

## Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regr = PolynomialFeatures(degree = 4)
year_poly = poly_regr.fit_transform(year)
poly_regr.fit(year_poly, mean)
lin_regr_2 = LinearRegression()
lin_regr_2.fit(year_poly, mean)

#Visualising the Linear Regression results
plt.scatter(year, mean, color= 'purple')
plt.plot(year, lin_regr.predict(year), color = 'blue', label = 'The Best Fit Line')
plt.title('year vs mean temperature (linear regression for GCAG)')
plt.xlabel('Years')
plt.ylabel('Mean Temperature')
plt.legend()
plt.show()

#Visualising the Polynomial Regression results
plt.scatter(year, mean, color= 'green')
plt.plot(year, lin_regr_2.predict(poly_regr.fit_transform(year)), color = 'blue', label = 'The Best Fit Line')
plt.title('year vs mean temperature (polynomial regression for GCAG)')
plt.xlabel('Years')
plt.ylabel('Mean Temperature')
plt.legend()
plt.show()

#Predicting a new result with Linear Regression
temp_2016 = lin_regr.predict([[2016]])
temp_2017 = lin_regr.predict([[2017]])

#Predicting a new result with Polynomial Regression
temp_poly_2016 = lin_regr_2.predict(poly_regr.fit_transform([[2016]]))
temp_poly_2017 = lin_regr_2.predict(poly_regr.fit_transform([[2017]]))

print("Temperatures for GCAG ")
print("A/c to linear regression, Temperature in 2016 will be: ", temp_2016)
print("A/c to polynomial regression, Temperature in 2016 will be: ", temp_poly_2016)
print("A/c to linear regression, Temperature in 2017 will be: ", temp_2017)
print("A/c to polynomial regression, Temperature in 2017 will be: ", temp_poly_2017)

#================================ FOR GISTEMP===================================
# Importing the libraries and the dataset for evaluation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('ann_temp_gistemp.csv')
year = dataset.iloc[:, 1:2].values
mean = dataset.iloc[:, 2:3].values

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regr = LinearRegression()
lin_regr.fit(year, mean)

## Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_regr = PolynomialFeatures(degree = 4)
year_poly = poly_regr.fit_transform(year)
poly_regr.fit(year_poly, mean)
lin_regr_2 = LinearRegression()
lin_regr_2.fit(year_poly, mean)

#Visualising the Linear Regression results
plt.scatter(year, mean, color= 'purple')
plt.plot(year, lin_regr.predict(year), color = 'blue', label = 'The Best Fit Line')
plt.title('year vs mean temperature (linear regression for GISTEMP)')
plt.xlabel('Years')
plt.ylabel('Mean Temperature')
plt.legend()
plt.show()

#Visualising the Polynomial Regression results
plt.scatter(year, mean, color= 'green')
plt.plot(year, lin_regr_2.predict(poly_regr.fit_transform(year)), color = 'blue', label = 'The Best Fit Line')
plt.title('year vs mean temperature (polynomial regression for GISTEMP)')
plt.xlabel('Years')
plt.ylabel('Mean Temperature')
plt.legend()
plt.show()

#Predicting a new result with Linear Regression
temp2016 = lin_regr.predict([[2016]])
temp2017 = lin_regr.predict([[2017]])

#Predicting a new result with Polynomial Regression
temp_poly2016 = lin_regr_2.predict(poly_regr.fit_transform([[2016]]))
temp_poly2017 = lin_regr_2.predict(poly_regr.fit_transform([[2017]]))

print("Temperatures for GISTEMP ")
print("A/c to linear regression, Temperature in 2016 will be: ", temp2016)
print("A/c to polynomial regression, Temperature in 2016 will be: ", temp_poly2016)
print("A/c to linear regression, Temperature in 2017 will be: ", temp2017)
print("A/c to polynomial regression, Temperature in 2017 will be: ", temp_poly2017)
