import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data
dataframe = pd.read_fwf("..\data\\brain_body.txt")
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]
x_train = x_values[0:30]
y_train = y_values[0:30]
x_test = x_values[31:62]
y_test = y_values[31:62]

# train the linear model
body_reg = linear_model.LinearRegression()
body_reg.fit(x_train, y_train)

# results
# The coefficients
print('Coefficients:', body_reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((body_reg.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % body_reg.score(x_test, y_test))

# visualize
plt.scatter(x_values, y_values,  color='black')
plt.xlabel("Brain Values")
plt.ylabel("Body Values")
plt.plot(x_test, body_reg.predict(x_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
