"""
https://cambridgespark.com/content/tutorials/from-simple-regression-to-multiple-regression-with-decision-trees/index.html
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt

bikes = pd.read_csv("..\data\\bikes.csv")

# setup test and training data
x_values = bikes[['temperature', 'humidity']]
y_values = bikes[['count']]
x_train = x_values[0:350]
y_train = y_values[0:350]
x_test = x_values[351:]
y_test = y_values[351:]

max_depth = 100

regressor = DecisionTreeRegressor(max_depth=max_depth)
regressor.fit(x_train, y_train)

DecisionTreeRegressor(criterion='mse', max_depth=max_depth, max_features=None,
           max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, presort=False, random_state=None,
           splitter='best')


# results
predicted_x_test = np.reshape(regressor.predict(x_test), (378, 1))

# The mean squared error
print("DecisionTreeRegressor Mean squared error: %.2f"
      % np.mean(np.sqrt((predicted_x_test - y_test) ** 2)))
sklearn_mse = mean_absolute_error(y_test, predicted_x_test)
print("Sklearn MSE: %.2f " % sklearn_mse)
# Explained variance score: 1 is perfect prediction
# http://blog.minitab.com/blog/adventures-in-statistics-2/regression-analysis-how-do-i-interpret-r-squared-and-assess-the-goodness-of-fit
print('DecisionTreeRegressor R-squared: %.2f' % regressor.score(x_test, y_test))

nx = 30
ny = 30
# creating a grid of points
x_temperature = np.linspace(-5, 40, nx) # min temperature -5, max 40
y_humidity = np.linspace(20, 80, ny) # min humidity 20, max 80
xx, yy = np.meshgrid(x_temperature, y_humidity)

# evaluating the regressor on all the points
z_bikes = regressor.predict(np.array([xx.flatten(), yy.flatten()]).T)
zz = np.reshape(z_bikes, (nx, ny))

fig = plt.figure(figsize=(8, 8))
# plotting the predictions
plt.pcolormesh(x_temperature, y_humidity, zz, cmap=plt.cm.YlOrRd)
plt.colorbar(label='bikes predicted') # add a colorbar on the right

# plotting also the observations
plt.scatter(x_test['temperature'], x_test['humidity'], s=y_test/25.0, c='g')

# setting the limit for each axis
plt.xlim(np.min(x_temperature), np.max(x_temperature))
plt.ylim(np.min(y_humidity), np.max(y_humidity))
plt.xlabel('temperature')
plt.ylabel('humidity')
plt.show()