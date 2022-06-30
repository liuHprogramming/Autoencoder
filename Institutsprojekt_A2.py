import numpy as np
import komm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses

import matplotlib.pyplot as plt

########################################   Frunktionen und Klassen   ###################################################

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='softmax'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(28, 28, 1)),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

################################################  Training  ###########################################################
#Loading the MNIST dataset in Keras
(train_images, train_labels), (train_images, train_labels) = mnist.load_data()
(test_images, test_labels), (test_images, test_labels) = mnist.load_data()


print(train_images.shape)
print(len(train_labels))
print(train_labels)

print(test_images.shape)
print(len(test_labels))
print(test_labels)


#Preparing the image data
#train_images = train_images.reshape((10000, 28*28))
train_images = train_images.astype("float32")/255
#test_images = train_images.reshape((10000, 28*28))
test_images = train_images.astype("float32")/255

train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

print (train_images.shape)
print (test_images.shape)

######################################     adding Noise to the pictures     ############################################

noise_factor = 0.2
train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape)
test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape)

train_images_noisy = tf.clip_by_value(train_images_noisy, clip_value_min=0., clip_value_max=1.)
test_images_noisy = tf.clip_by_value(test_images_noisy, clip_value_min=0., clip_value_max=1.)


######################################    Encoding and Decoding Images      ############################################

autoencoder = Denoise()   #Noise entfernen

#Compilation
autoencoder.compile(optimizer='rmsprop',
                    loss=losses.MeanSquaredError(),
                    metrics = ["accuracy"])

autoencoder.fit(train_images, train_images,
                epochs=5,
                shuffle=True,
                validation_data=(train_images, train_images))

encoded_imgs = autoencoder.encoder(train_images).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


autoencoder.encoder.summary()
autoencoder.decoder.summary()

# encoding
encoded_imgs = autoencoder.encoder(train_images_noisy).numpy()

# AWGN im Channel addieren
awgn = komm.AWGNChannel()
awgn.snr = 7.5
for img in encoded_imgs:            # Tensor aus jedem Bild der Datenbank aufrufen
    for mat in img:                 # jede 7*8 Matrix aus dem Tensor aufrufen
        for arr in mat:             # jede 1*8 Array aus der Matrix aufrufen
            arr = awgn(arr)         # addieren AWGN in jedem Array

# Tensor des ersten Bildes aus der Datenbank ausgeben,nachdem wir AWGN im Channel addiert haben
print(encoded_imgs[0])

# decoding
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    #Display original mit noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(train_images_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #Display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()