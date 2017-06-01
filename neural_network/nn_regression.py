import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston

# load dataset
dataframe = pandas.read_csv("..\data\housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# boston = load_boston() #Directly load from sklearn

# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]

print(X)