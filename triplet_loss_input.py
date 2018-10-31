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


def build_model(img_x, img_y, output = 0):
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
    if output:
        ### binary output
        hash_fc = Dense(50,
                        activation=Lambda(lambda z: tf.divide(
                            tf.add(
                                K.sign(
                                    tf.subtract(keras.layers.activations.sigmoid(x=z), 0.5)),
                                K.abs(
                                    K.sign(
                                        tf.subtract(keras.layers.activations.sigmoid(x=z), 0.5)))),
                            2)), kernel_initializer="lecun_normal")(merged_fc)
    else:
        ### Sigmoid output
        hash_fc = Dense(50,
                        activation="sigmoid",
                        kernel_initializer="lecun_normal")(merged_fc)

    model = Model(inputs=[input_shape], outputs=[hash_fc])
    return model

def generate_input_data(data_type = 0, output = 0):
    if data_type:
        x_train, x_labels = data_reader.get_validation_set()
    else:
        x_train, x_labels = data_reader.get_training_set()

    batch_size = 128
    epochs = 10
    img_x, img_y = 60, 160
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
    input_shape = (img_x, img_y, 3)
    x_train = x_train.astype('float32')
    x_train /= 255

    model = build_model(img_x, img_y, output)
    r = model.predict(x_train,
                    batch_size=batch_size,
                    verbose=1)
    #print(r[0])

    #Save output to CSV
    if data_type:
        np.savetxt("outputs_for_validation.csv", r, delimiter=",")
    else:
        np.savetxt("outputs_for_mining.csv", r , delimiter=",")





