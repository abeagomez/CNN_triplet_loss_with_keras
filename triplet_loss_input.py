from __future__ import print_function
import data_reader
import cv2
import random
import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda, merge, concatenate
from keras.models import Sequential
from keras import Input, Model
import keras.backend as K
import matplotlib.pylab as plt
import tensorflow as tf


#x_train, x_labels = data_reader.get_training_set()
x_train, x_labels = data_reader.get_validation_set()

batch_size = 100
epochs = 10

img_x, img_y = 60, 160

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
input_shape = (img_x, img_y, 3)
x_train = x_train.astype('float32')
x_train /= 255

### Sequential Model #####################################################
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu'))
# model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
# model.add(Flatten())
# model.add(Dense(4024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512,
#                 activation= Lambda(lambda z: K.round(
#                 keras.layers.activations.sigmoid(x=z))),
#                 kernel_initializer="lecun_normal"))
##########################################################################
input_shape = Input(shape=(img_x, img_y, 3))
conv_0 = Conv2D(32, kernel_size=(3, 3), strides=(
    1, 1), activation='relu')(input_shape)
max_p0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_0)
conv_1 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(max_p0)
max_p1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_1)
conv_2 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(max_p1)
max_p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_2)
conv_3 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(max_p2)
max_p3 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(conv_3)
flatten = Flatten()(max_p3)
dense1 = Dense(4024, activation='relu')(flatten)
dense2 = Dense(512, activation='relu')(dense1)
merged_fc = concatenate([dense1, dense2])
#binary output
hash_fc = Dense(50,
                activation=Lambda(lambda z: tf.divide(
                    tf.add(
                        K.sign(
                            tf.subtract(keras.layers.activations.sigmoid(x=z), 0.5)),
                        K.abs(
                            K.sign(
                                tf.subtract(keras.layers.activations.sigmoid(x=z), 0.5)))),
                    2)), kernel_initializer="lecun_normal")(merged_fc)
#hash_fc = Dense(50,
#            activation=Lambda(lambda z: K.round(keras.layers.activations.sigmoid(x=z))),
#            kernel_initializer="lecun_normal")(merged_fc)
#Sigmoid output
#hash_fc = Dense(50,
#            activation="sigmoid",
#            kernel_initializer="lecun_normal")(merged_fc)
model = Model(inputs=[input_shape], outputs=[hash_fc])


#Testing fragment
# while(epochs > 6):
#     r = model.predict(x_train,
#                         batch_size=batch_size,
#                         verbose=1)
#     #r = model.train_on_batch(x_train, x_labels)
#     print(r[0])
#     epochs -= 1

r = model.predict(x_train,
                  batch_size=batch_size,
                  verbose=1)
#print(r[0])

#Save output to CSV
#np.savetxt("outputs_for_mining.csv", r , delimiter=",")
np.savetxt("outputs_for_validation.csv", r, delimiter=",")




