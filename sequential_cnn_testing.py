from __future__ import print_function
import cv2
import random
import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras import Input, Model
import keras.backend as K
import matplotlib.pylab as plt

x_train, x_labels = read_data("./labeled/train/")
x_test, y_test = x_train, x_labels

batch_size=5
num_classes=100
epochs=1

img_x, img_y = 128, 254

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
input_shape = (img_x, img_y, 3)
x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 3)
x_test = x_train.astype('float32')
x_test /= 255
x_labels = keras.utils.to_categorical(x_labels, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu'))
#model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
model.add(Flatten())
model.add(Dense(4024, activation='relu'))
model.add(Dense(512, activation='sigmoid'))
#model.add(Dense(num_classes, activation='softmax'))

output = model(K.variable(x_train))
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adam(),
#              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()

# model.fit(x_train, x_labels,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test),
#           callbacks=[history])
print(len(model.predict(x_train,
               batch_size=batch_size,
               verbose=1)[0]))

#Save output to CSV
#np.savetxt("foo.csv", K.eval(output) , delimiter=",")

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#plt.plot(range(1, 11), history.acc)
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.show()
