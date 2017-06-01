import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense
import numpy

"""
 Tutorial from: http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
"""

# fix random seed for reproducibility
numpy.random.seed(7)

"""
Load Data
"""
# load pima indians dataset
dataset = numpy.loadtxt("..\data\pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

"""
Define Model

The first layer has 12 neurons and expects 8 input variables. 
The second hidden layer has 8 neurons and finally, 
the output layer has 1 neuron to predict the class
"""
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

"""
Compile Model

Compiling the model uses the efficient numerical libraries under the covers (the so-called backend) 
such as Theano or TensorFlow. The backend automatically chooses the best way to represent the network 
for training and making predictions to run on your hardware, such as CPU or GPU or even distributed.

We must specify the loss function to use to evaluate a set of weights, 
the optimizer used to search through different weights for the network 
and any optional metrics we would like to collect and report during training.
"""
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

"""
 Fit Model
 
 The training process will run for a fixed number of iterations through the dataset called epochs, 
 that we must specify using the nepochs argument. We can also set the number of instances that are 
 evaluated before a weight update in the network is performed, 
 called the batch size and set using the batch_size argument.
"""
model.fit(X, Y, epochs=150, batch_size=10)

"""
Evaluate Model
"""
scores = model.evaluate(X, Y)

i = 0
print("\nEvaluation Results:")
for metric_name in model.metrics_names:
    print("\n%s: %.2f%%" % (metric_name, scores[i] * 100))
    i += 1
