import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

bikes = pd.read_csv("..\data\\bikes.csv")

print(bikes.head())

# setup test and training data
x_values = bikes[['temperature']]
y_values = bikes[['count']]
x_train = x_values[0:350]
y_train = y_values[0:350]
x_test = x_values[351:]
y_test = y_values[351:]

"""
Decision tree model
"""

dec_tree = DecisionTreeRegressor(max_depth=2)
dec_tree.fit(x_train, y_train)

export_graphviz(dec_tree, out_file='lr_bikes_visualize.dot', feature_names=['temperature'])

predicted_x_test = np.reshape(dec_tree.predict(x_test), (378, 1))

# results
# The mean squared error
print("DecisionTreeRegressor Mean squared error: %.2f"
      % np.mean(np.sqrt((predicted_x_test - y_test) ** 2)))
sklearn_mse = mean_absolute_error(y_test, predicted_x_test)
print("DecisionTreeRegressor sklearn MSE: %.2f " % sklearn_mse)
# Explained variance score: 1 is perfect prediction
# http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit
print('DecisionTreeRegressor R-squared: %.2f' % dec_tree.score(x_test, y_test))

xx = np.array([np.linspace(-5, 40, 100)]).T

plt.figure(figsize=(8,6))
plt.plot(bikes['temperature'], bikes['count'], 'o', label='observation')
plt.plot(xx, dec_tree.predict(xx), linewidth=4, alpha=.7, label='prediction')
plt.xlabel('temperature')
plt.ylabel('bike count')
plt.legend()
# plt.show()

# sys.exit(0)

"""
Linear regression model
"""

# implement linear regression model
bikes_reg = linear_model.LinearRegression()
bikes_reg.fit(x_train, y_train)

# The coefficients: y = m*x + c
print('LinearRegression Coefficients:', bikes_reg.coef_)
# The mean squared error
print("LinearRegression Mean squared error: %.2f"
      % np.mean(np.sqrt((bikes_reg.predict(x_test) - y_test) ** 2)))
sklearn_mse_LR = mean_absolute_error(y_test, bikes_reg.predict(x_test))
print("LinearRegression sklearn MSE: %.2f " % sklearn_mse_LR)
# Explained variance score: 1 is perfect prediction
print('LinearRegression R-squared: %.2f' % bikes_reg.score(x_test, y_test))

# visualize
plt.scatter(x_values, y_values,  color='black')
plt.xlabel('temperature')
plt.ylabel('bike count')
plt.plot(x_test, bikes_reg.predict(x_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

"""
Understand the correlation of the parameters.
We can clearly see only temperature has a correlation with the bikes count.
"""

# print("Correlation temperature:", round(stats.pearsonr(bikes['temperature'], bikes['count'])[0], 2))
# print("Correlation humidity:", round(stats.pearsonr(bikes['humidity'], bikes['count'])[0], 2))
# print("Correlation windspeed:", round(stats.pearsonr(bikes['windspeed'], bikes['count'])[0], 2))
#
# plt.figure(1)
# plt.plot(bikes['temperature'], bikes['count'], 'o')
# plt.xlabel('Temperature')
# plt.ylabel('Bikes Count')
# plt.figure(2)
# plt.plot(bikes['humidity'], bikes['count'], 'o')
# plt.xlabel('humidity')
# plt.ylabel('Bikes Count')
# plt.figure(3)
# plt.plot(bikes['windspeed'], bikes['count'], 'o')
# plt.xlabel('windspeed')
# plt.ylabel('Bikes Count')
# plt.show()