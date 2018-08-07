'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

MXNET
25000/25000 [==============================] - 431s 17ms/step - loss: 0.2289 - acc: 0.9089 - val_loss: 0.2582 - val_acc: 0.8943
Training time 810.3772482872009 (s)
Evaluation time 96.83182692527771 (s)
Test score: 0.25815799966573716
Test accuracy: 0.89428

Tensorflow
25000/25000 [==============================] - 14s 555us/step - loss: 0.2298 - acc: 0.9074 - val_loss: 0.2819 - val_acc: 0.8821
Training time 34.33080840110779 (s)
Evaluation time 2.610790252685547 (s)
Test score: 0.28191687469005583
Test accuracy: 0.88208
'''
from __future__ import print_function

"""
https://medium.com/apache-mxnet/keras-gets-a-speedy-new-backend-with-keras-mxnet-3a853efc1d75
"""

import os
import time
# os.environ['KERAS_BACKEND'] = 'mxnet'

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb


# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
print("Training time", (time.time() - start), '(s)')

start = time.time()
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print("Evaluation time", (time.time() - start), '(s)')

print('Test score:', score)
print('Test accuracy:', acc)
