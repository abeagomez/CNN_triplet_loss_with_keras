import keras
from keras import metrics
import keras.backend as K
import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, merge, concatenate
from keras.models import Sequential
import triplets_mining
from keras.layers import Lambda
import numpy as np
import os

def triplet_loss(y):
    anchor, positive, negative = tf.split(y, 3, axis = 1)
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), 0.05)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss

def build_model(img_x, img_y):
    input_shape = Input(shape=(img_x, img_y, 3))

    conv_0 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu') (input_shape)
    max_p0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (conv_0)
    conv_1 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu') (max_p0)
    max_p1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_1)
    conv_2 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu') (max_p1)
    max_p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (conv_2)
    conv_3 = Conv2D(32, (3, 3), strides=(1, 1), activation='relu') (max_p2)
    max_p3 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1)) (conv_3)
    flatten = Flatten() (max_p2)
    dense1 = Dense(4024, activation='relu')(flatten)
    dense2 = Dense(512, activation='sigmoid')(dense1)

    anchor = Input(shape=(128, 254, 3))
    positive = Input(shape=(128, 254, 3))
    negative = Input(shape=(128, 254, 3))

    reid_model = Model(inputs=[input_shape], outputs=[dense2])

    anchor_embed = reid_model(anchor)
    positive_embed = reid_model(positive)
    negative_embed = reid_model(negative)

    merged_output = concatenate([anchor_embed, positive_embed, negative_embed])
    loss = Lambda(triplet_loss, (1,))(merged_output)

    model = Model(inputs=[anchor, positive, negative], outputs=loss)
    model.compile(optimizer='Adam', loss='mse',
                  metrics=["mae"])
    return model


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


class CollectWeightCallback(keras.callbacks.Callback):
    def __init__(self, layer_index):
        super(CollectWeightCallback, self).__init__()
        self.layer_index = layer_index
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        layer = self.model.layers[self.layer_index]
        self.weights.append(layer.get_weights())


num_epochs = 10
img_x, img_y = 60, 160
#Esto de abajo son 3 arrays de numpy que representan imagenes RGB
#Cada posicion es una imagen RGB de 128(ancho)x254(alto)
x_anchor, x_positive, x_negative = triplets_mining.get_hard_triplets(5000,0)
l = len(x_anchor)
x_anchor = x_anchor.reshape(x_anchor.shape[0], img_x, img_y, 3)
x_anchor = x_anchor.astype('float32')
x_anchor /= 255

x_positive = x_positive.reshape(x_positive.shape[0], img_x, img_y, 3)
x_positive = x_positive.astype('float32')
x_positive /= 255

x_negative = x_negative.reshape(x_negative.shape[0], img_x, img_y, 3)
x_negative = x_negative.astype('float32')
x_negative /= 255

xt_anchor, xt_positive, xt_negative = triplets_mining.get_valid_validation_triplets(500, 0)
lt = len(xt_anchor)
xt_anchor = xt_anchor.reshape(xt_anchor.shape[0], img_x, img_y, 3)
xt_anchor = xt_anchor.astype('float32')
xt_anchor /= 255

xt_positive = xt_positive.reshape(xt_positive.shape[0], img_x, img_y, 3)
xt_positive = xt_positive.astype('float32')
xt_positive /= 255

xt_negative = xt_negative.reshape(xt_negative.shape[0], img_x, img_y, 3)
xt_negative = xt_negative.astype('float32')
xt_negative /= 255

x = [x_anchor, x_positive, x_negative]
x_test = [xt_anchor, xt_positive, xt_negative]
cnn_model = build_model(img_x, img_y)
# Print the model structure
print(cnn_model.summary())

history = AccuracyHistory()
cnn_model.fit(x=x,y = np.zeros(l),
            batch_size=64,
            epochs=10,
            verbose=1,
            validation_data=(x_test, np.zeros(lt)),
            callbacks=[history])

score = cnn_model.evaluate(x = x_test, y = np.zeros(lt),verbose = 0)
print('Test loss:', score[0])
#print('Test accuracy:', score[1])

# serialize model to JSON
#model_json = cnn_model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)

# serialize weights to HDF5
#cnn_model.save_weights("model.h5")
#print("Saved model to disk")

# for epoch in range(num_epochs):
#     print('Epoch %s' % epoch)

#     model.fit([x_anchor, x_positive, x_negative],
#             y=np.zeros(l),
#             batch_size=64,
#             epochs=1,
#             verbose=1,
#             callbacks=[history])
#     # all layer outputs
#     outputs = model.layers[-1].output
#     #functor = K.function([inp, K.learning_phase()], outputs)
#     print(outputs)
