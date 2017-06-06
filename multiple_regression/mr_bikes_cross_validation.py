"""
https://cambridgespark.com/content/tutorials/from-simple-regression-to-multiple-regression-with-decision-trees/index.html
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import cross_val_score
import numpy as np
from matplotlib import pyplot as plt

bikes = pd.read_csv("..\data\\bikes.csv")

# setup test and training data
x_values = bikes[['temperature', 'humidity']]
y_values = bikes[['count']]

max_depth = 100

regressor = DecisionTreeRegressor(max_depth=max_depth)
regressor.fit(x_values, y_values)

DecisionTreeRegressor(criterion='mse', max_depth=max_depth, max_features=None,
           max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, presort=False, random_state=None,
           splitter='best')


# results
scores = -cross_val_score(regressor, bikes[['temperature', 'humidity']],
bikes['count'], scoring='neg_mean_absolute_error', cv=10)
print("Cross validation mean absolute error %.2f" % scores.mean())

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
plt.scatter(bikes['temperature'], bikes['humidity'], s=y_values/25.0, c='g')

# setting the limit for each axis
plt.xlim(np.min(x_temperature), np.max(x_temperature))
plt.ylim(np.min(y_humidity), np.max(y_humidity))
plt.xlabel('temperature')
plt.ylabel('humidity')
plt.show()