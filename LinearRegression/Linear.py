# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# type this if array isn't showing
np.set_printoptions(threshold=np.nan)

# splitting dataset into train_set and test_set
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

'''
# feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
'''

# fitting Simple Linear Regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the Test_set results
y_pred = regressor.predict(X_test)

# visualizing the Training_set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color='blue') # X_pred = regressor.predict(X_train)
plt.title('Experience vs $Salary')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# visualizing the test_set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Experience vs Salary(test)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
