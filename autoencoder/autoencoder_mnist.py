from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
To build an autoencoder, you need three things: 
1. an encoding function, 
2. a decoding function, and 
3. a distance function between the amount of information loss between the compressed representation of your data 
and the decompressed representation (i.e. a "loss" function). 
The encoder and decoder will be chosen to be parametric functions (typically neural networks), and to be differentiable 
with respect to the distance function, so the parameters of the  encoding/decoding functions can be optimize to minimize 
the reconstruction loss, using Stochastic Gradient Descent.

Today two interesting practical applications of autoencoders are 
1. data denoising, and 
2. dimensionality reduction for data visualization. 
With appropriate dimensionality and sparsity constraints, autoencoders can learn data projections that are more 
interesting than PCA or other basic techniques.

Autoencoders are not a true unsupervised learning technique, they are a self-supervised technique, 
a specific instance of supervised learning where the targets are generated from the input data.
"""

#  We're using MNIST digits, and
# we're discarding the labels (since we're only interested in encoding/decoding the input images)
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print("Train data shape:", x_train.shape)
print("Test data shape:", x_test.shape)


# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats ( 784 / 32 = 24.5 )

# this is our input placeholder
input_img = Input(shape=(784,))

"""
input_img -> encorder (Model) -> encoded (Dense) -> decorder (Model) -> decoded (Dense)
"""

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # # display encoded
    # ax = plt.subplot(2, n, i + 1 + n)
    # plt.imshow(encoded_imgs[i].reshape(8, 8))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
