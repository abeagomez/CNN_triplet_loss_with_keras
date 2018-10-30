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


def structured_triplet_loss(y):
    anchor, positive, anchor_neg, positive_neg = tf.split(y, 4, axis=1)
    #anchor, positive = tf.round(anchor), tf.round(positive)
    #anchor_neg, positive_neg = tf.round(anchor_neg), tf.round(positive_neg)
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist_anch = tf.reduce_sum(tf.square(tf.subtract(anchor, anchor_neg)), 1)
    neg_dist_pos = tf.reduce_sum(tf.square(tf.subtract(anchor, positive_neg)), 1)

    term_anchor = tf.maximum(0.0, tf.subtract(1.0, neg_dist_anch))
    term_positive = tf.maximum(0.0, tf.subtract(1.0, neg_dist_pos))
    inner_max = tf.maximum(term_anchor, term_positive)
    basic_loss = tf.maximum(tf.add(inner_max, pos_dist), 0.0)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss

def build_model(img_x, img_y):
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
    dense2 = Dense(512, activation='sigmoid')(dense1)
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
    #hash_fc = Dense(50, activation = "sigmoid")(merged_fc)

    anchor = Input(shape=(60, 160, 3))
    positive = Input(shape=(60, 160, 3))
    a_negative = Input(shape=(60, 160, 3))
    p_negative = Input(shape=(60, 160, 3))

    reid_model = Model(inputs=[input_shape], outputs=[hash_fc])

    anchor_embed = reid_model(anchor)
    positive_embed = reid_model(positive)
    a_negative_embed = reid_model(a_negative)
    p_negative_embed = reid_model(p_negative)

    merged_output = concatenate(
        [anchor_embed, positive_embed, a_negative_embed, p_negative_embed])
    loss = Lambda(structured_triplet_loss, (1,))(merged_output)

    model = Model(inputs=[anchor, positive, a_negative, p_negative], outputs=loss)
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

def training_set(img_x, img_y, no_triplets, data_type):
    #Esto de abajo son 4 arrays de numpy que representan imagenes RGB
    #Cada posicion es una imagen RGB de 60(ancho)x160(alto)
    x_anchor, x_positive, x_anchor_neg, x_pos_neg = triplets_mining.get_structured_hard_triplets(
        no_triplets, data_type)
    l = len(x_anchor)
    x_anchor = x_anchor.reshape(x_anchor.shape[0], img_x, img_y, 3)
    x_anchor = x_anchor.astype('float32')
    x_anchor /= 255

    x_positive = x_positive.reshape(x_positive.shape[0], img_x, img_y, 3)
    x_positive = x_positive.astype('float32')
    x_positive /= 255

    x_a_negative = x_anchor_neg.reshape(x_anchor_neg.shape[0], img_x, img_y, 3)
    x_a_negative = x_a_negative.astype('float32')
    x_a_negative /= 255

    x_p_negative = x_pos_neg.reshape(x_pos_neg.shape[0], img_x, img_y, 3)
    x_p_negative = x_p_negative.astype('float32')
    x_p_negative /= 255

    return l, [x_anchor, x_positive, x_a_negative, x_p_negative]

def validation_set(img_x, img_y, no_triplets, data_type):
    xt_anchor,xt_positive,xt_a_negative,xt_p_negative = triplets_mining.get_structured_validation_triplets(
    no_triplets, data_type)
    lt = len(xt_anchor)
    xt_anchor = xt_anchor.reshape(xt_anchor.shape[0], img_x, img_y, 3)
    xt_anchor = xt_anchor.astype('float32')
    xt_anchor /= 255

    xt_positive = xt_positive.reshape(xt_positive.shape[0], img_x, img_y, 3)
    xt_positive = xt_positive.astype('float32')
    xt_positive /= 255

    xt_a_negative = xt_a_negative.reshape(xt_a_negative.shape[0], img_x, img_y, 3)
    xt_a_negative = xt_a_negative.astype('float32')
    xt_a_negative /= 255

    xt_p_negative = xt_p_negative.reshape(xt_p_negative.shape[0], img_x, img_y, 3)
    xt_p_negative = xt_p_negative.astype('float32')
    xt_p_negative /= 255

    return lt, [xt_anchor, xt_positive, xt_a_negative, xt_p_negative]


num_epochs = 10
img_x, img_y = 60, 160
l, x = training_set(img_x, img_y, 7000, 0)
lt, x_test = validation_set(img_x, img_y, 900, 0)
cnn_model = build_model(img_x, img_y)
# Print the model structure
print(cnn_model.summary())

history = AccuracyHistory()
cnn_model.fit(x=x, y=np.zeros(l),
              batch_size=64,
              epochs=10,
              verbose=1,
              validation_data=(x_test, np.zeros(lt)),
              callbacks=[history])

score = cnn_model.evaluate(x=x_test, y = np.zeros(lt), verbose=0)
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

#     cnn_model.fit(x,
#             y=np.zeros(l),
#             batch_size=64,
#             epochs=1,
#             verbose=1,
#             callbacks=[history])
#     # all layer outputs
#     outputs = model.layers[-1].output
#     #functor = K.function([inp, K.learning_phase()], outputs)
#     print(outputs)
    # r = cnn_model.predict(x,
    #                   batch_size=64,
    #                   verbose=1)
    # print(r)
