# polynomial regression
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# no need of splitting because of lesser data and to predict best(accurate) result
# fitting Linear Regression on the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# fitting the dataset to Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # degree = 4 for more accurate model
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# visualizing the Linear Regression Model
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth or Bluff (Linear Model)")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()

# visualizing the Polynomial Regression model
'''
# for making the curve more precise
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
'''

plt.scatter(X, y, color = "red")
# arg(X, lin_reg_2.predict(poly_reg.fit_transform(X)))
plt.plot(X, lin_reg_2.predict(X_poly), color = "blue") 
plt.title("Truth or Bluff (Polynomial Model)")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()

# predicting the result with Linear Regression Model
lin_reg.predict(6.5)

# predicting with Polynomial Regression Model
lin_reg_2.predict(poly_reg.fit_transform(6.5))
