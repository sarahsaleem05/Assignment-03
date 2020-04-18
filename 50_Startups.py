# SARAH SALEEM AKHTER
# Using Decision Tree Regression
# 50 Startups
# Taking data of Florida and NewYork

# Importing Libraries 
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset for Florida
dataset = pd.read_csv('50_Startups_Flor.csv')
v1 = dataset.iloc[:, 0:1].values #R&D Spend
v2 = dataset.iloc[:, 1:2].values #Administration
v3 = dataset.iloc[:, 2:3].values #Marketing Spend
sum = v1 + v2 + v3
profit = dataset.iloc[:, 4].values #Profit Generated

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
sum_train, sum_test, profit_train, profit_test = train_test_split(sum, profit, test_size = 0.2, random_state = 0)

#Fitting Florida's dataset by Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor 
RegrVaria = DecisionTreeRegressor(random_state = 0)
RegrVaria.fit (sum , profit)

# Predicting new result for Florida
Pred_Flor = RegrVaria.predict ([[8000000]])

plt.scatter(sum, profit, color = 'blue')
plt.plot(sum, RegrVaria.predict(sum), color = 'green', label = 'Best Fit Line')
plt.title('Cost Vs Profit')
plt.xlabel('Cost')
plt.ylabel('Profit')
plt.legend()
plt.show()


#Importing dataset for NewYork
dataset = pd.read_csv('50_Startups_NY.csv')
v1 = dataset.iloc[:, 0:1].values #R&D Spend
v2 = dataset.iloc[:, 1:2].values #Administration
v3 = dataset.iloc[:, 2:3].values #Marketing Spend
sum = v1 + v2 + v3
profit = dataset.iloc[:, 4].values #Profit Generated

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
sum_train, sum_test, profit_train, profit_test = train_test_split(sum, profit, test_size = 0.2, random_state = 0)

#Fitting NewYork's dataset by Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor 
RegrVaria = DecisionTreeRegressor(random_state = 0)
RegrVaria.fit (sum , profit)

# Predicting new result for NewYork
Pred_NY = RegrVaria.predict ([[8000000]])

plt.scatter(sum, profit, color = 'brown')
plt.plot(sum, RegrVaria.predict(sum), color = 'grey', label = 'Best Fit Line')
plt.title('Cost Vs Profit')
plt.xlabel('Cost')
plt.ylabel('Profit')
plt.legend()
plt.show()

print("Predicted Profit for FLorida: ", Pred_Flor)
print("Predicted Profit for NewYork: ", Pred_NY)
 