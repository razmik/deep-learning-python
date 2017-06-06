"""
http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy, pandas, time
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# load dataset
dataframe = pandas.read_csv("..\data\housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]

# define base model
"""
simple model that has a single fully connected hidden layer with the same number of neurons as input attributes (13). 
The network uses good practices such as the rectifier activation function for the hidden layer. 
No activation function is used for the output layer because it is a regression problem and we are interested in 
predicting numerical values directly without transform.
"""
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
start = time.time()
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
end = time.time()
print("Basic Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
print("Basic Training Duration: %.4f seconds \n" % (end-start))

"""
Modeling The Standardized Dataset

It is almost always good practice to prepare your data before modeling it using a neural network model.

Continuing on from the above baseline model, we can re-evaluate the same model using a standardized version of 
the input dataset. We can use scikit-learn’s Pipeline framework to perform the standardization during the model 
evaluation process, within each fold of the cross validation. This ensures that there is no data leakage from 
each testset cross validation fold into the training data.
"""
# evaluate model with standardized dataset
numpy.random.seed(seed)
estimators = []
start = time.time()
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
end = time.time()
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
print("Standardized Training Duration: %.4f seconds \n" % (end-start))


"""
Evaluate a Deeper Network Topology

One way to improve the performance a neural network is to add more layers. This might allow the model to extract 
and recombine higher order features embedded in the data.
"""

# define the model
def larger_model():
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
start = time.time()
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
end = time.time()
print("Larger (input -> 13 hidden -> 6 hidden -> output): %.2f (%.2f) MSE" % (results.mean(), results.std()))
print("Larger Training Duration: %.4f seconds \n" % (end-start))

"""
Evaluate a Wider Network Topology
"""

# define wider model
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

numpy.random.seed(seed)
start = time.time()
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
end = time.time()
print("Wider (Input -> 20 -> output): %.2f (%.2f) MSE" % (results.mean(), results.std()))
print("Wider Training Duration: %.4f seconds \n" % (end-start))