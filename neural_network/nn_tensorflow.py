# code: https://github.com/llSourcell/How_to_use_Tensorflow_for_classification-LIVE/blob/master/demo.ipynb

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

dataframe = pd.read_csv('../data/houses.csv')
dataframe = dataframe.drop(['index', 'price', 'sq_price'], axis=1)
dataframe = dataframe[0:10]

dataframe.loc[:, ('y1')] = [1,1,1,0,0,1,0,1,1,1]
dataframe.loc[:, ('y2')] = dataframe['y1'] == 0
dataframe.loc[:, ('y2')] = dataframe['y2'].astype(int)

"""
Step 3:
Now that we have all our data in the dataframe, we'll need to shape it in matrices to feed it to TensorFlow
Prepare data for tensorflow (tensor)
tensors are generic verison of vectors and matrices
vector is list of numbers (1D)
matrics of list of list of numbers (2D)
list of list of list of numbers (3D)

Convert features to input tensors
"""

inputX = dataframe.loc[:, ['area', 'bathrooms']].as_matrix()
inputY = dataframe.loc[:, ["y1", "y2"]].as_matrix()

# Step 4: Write our hyperparameters

learning_rate = 0.000001
training_epochs = 2000
display_step = 50
n_samples = inputY.size

# Step 5: Create our computational graph

x = tf.placeholder(tf.float32, [None, 2])   # Okay TensorFlow, we'll feed you an array of examples. Each example will
                                            # be an array of two float values (area, and number of bathrooms).
                                            # "None" means we can feed you any number of examples
                                            # Notice we haven't fed it the values yet

W = tf.Variable(tf.zeros([2,2]))

b = tf.Variable(tf.zeros([2]))              # Also maintain two bias values

y_values = tf.add(tf.matmul(x, W), b)

#softmax = sigmoid
y = tf.nn.softmax(y_values)

y_target = tf.placeholder(tf.float32, [None, 2])


#Step 6: training
# create our cost function - mean squarred error
cost = tf.reduce_sum(tf.pow(y_target - y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(training_epochs):

    sess.run(optimizer, feed_dict={x: inputX, y_target: inputY})

    if i % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_target: inputY})
        print("Training step:", '%04d' % i, "cost=", "{:.9f}".format(cc))  # , \"W=", sess.run(W), "b=", sess.run(b)

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={x: inputX, y_target: inputY})
print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
