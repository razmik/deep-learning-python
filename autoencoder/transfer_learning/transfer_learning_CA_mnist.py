from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import TensorBoard
from keras.models import Model, model_from_json, Sequential
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def save_model(encoder, autoencoder):
    # serialize encoder to JSON
    model_json = encoder.to_json()
    with open("encoder.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    encoder.save_weights("encoder_weights.h5")
    print("Saved encoder to disk")

    # serialize autoencoder to JSON
    model_json = autoencoder.to_json()
    with open("autoencoder.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    autoencoder.save_weights("autoencoder_weights.h5")
    print("Saved autoencoder to disk")


def load_model():
    # load json and create model
    fname = 'encoder.json'
    json_file = open(fname, 'r')
    loaded_encoder_json = json_file.read()
    json_file.close()
    loaded_encoder = model_from_json(loaded_encoder_json)
    # load weights into new model
    loaded_encoder.load_weights("encoder_weights.h5")
    print("Loaded encoder from disk")

    # load json and create model
    json_file = open('autoencoder.json', 'r')
    loaded_autoencoder_json = json_file.read()
    json_file.close()
    loaded_autoencoder = model_from_json(loaded_autoencoder_json)
    # load weights into new model
    loaded_autoencoder.load_weights("autoencoder_weights.h5")
    print("Loaded autoencoder from disk")

    return loaded_encoder, loaded_autoencoder


def display(x_test, encoded_imgs, decoded_imgs, plt_num, plt_title, n=10):

    plt.figure(plt_num, figsize=(20, 4))
    plt.title(plt_title)
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display encoded
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(encoded_imgs[i].reshape(8, 16))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


"""
Stack several layers of hidden layers.
"""

#  We're using MNIST digits, and
# we're discarding the labels (since we're only interested in encoding/decoding the input images)
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

x_train_phase_1 = x_train[:int(len(x_train)/2)]
x_train_phase_2 = x_train[int(len(x_train)/2):]


mode = 2  # 1= train, 2= transfer learn

if mode == 1:
    x_train = x_train[:int(len(x_train)/2)]
    print("Train data shape:", x_train.shape)
    print("Test data shape:", x_test.shape)

    # this is our input placeholder
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    autoencoder.summary()

    # First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=2,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    display(x_test, encoded_imgs, decoded_imgs, 1, plt_title='1- Pre after initial training')

    save_model(encoder, autoencoder)

elif mode == 2:
    x_train = x_train[int(len(x_train)/2):]
    print("Train data shape:", x_train.shape)
    print("Test data shape:", x_test.shape)

    encoder, autoencoder = load_model()

    autoencoder.summary()

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    display(x_test, encoded_imgs, decoded_imgs, 2, plt_title='2 - Pred just after loading')

    # First, we'll configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer:
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=30,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    display(x_test, encoded_imgs, decoded_imgs, 3, plt_title='3 - Pred after transfer learning')


plt.show()


