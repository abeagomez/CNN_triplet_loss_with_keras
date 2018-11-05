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
import os, triplet_loss_input
from loading_weights import get_model_output, build_dict


def multi_input_triplet_loss(x, y):
    return triplet_loss(y)

def triplet_loss(y):
    anchor, positive, negative = tf.split(y, 3, axis=1)

    #Original loss
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), 0.05)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss

def structured_triplet_loss(x, y):
    anchor, positive, anchor_neg, positive_neg = tf.split(y, 4, axis=1)

    # Simple approach
    # pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # neg_dist_anch = tf.reduce_sum(
    #     tf.square(tf.subtract(anchor, anchor_neg)), 1)
    # neg_dist_pos = tf.reduce_sum(
    #     tf.square(tf.subtract(anchor, positive_neg)), 1)

    # term_anchor = tf.maximum(0.0, tf.subtract(1.0, neg_dist_anch))
    # term_positive = tf.maximum(0.0, tf.subtract(1.0, neg_dist_pos))
    # inner_max = tf.maximum(term_anchor, term_positive)
    # basic_loss = tf.maximum(tf.add(inner_max, pos_dist), 0.0)
    # loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    # l2 norm
    pos_dist = tf.reduce_sum(tf.square(tf.norm(tf.subtract(anchor, positive))), 1)
    neg_dist_anch = tf.reduce_sum(tf.square(tf.norm(tf.subtract(anchor, anchor_neg))), 1)
    neg_dist_pos = tf.reduce_sum(tf.square(tf.norm(tf.subtract(anchor, positive_neg))), 1)

    term_anchor = tf.maximum(0.0, tf.subtract(1.0, neg_dist_anch))
    term_positive = tf.maximum(0.0, tf.subtract(1.0, neg_dist_pos))
    inner_max = tf.maximum(term_anchor, term_positive)
    basic_loss = tf.maximum(tf.add(inner_max, pos_dist), 0.0)
    # loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return basic_loss

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def build_model(img_x, img_y, output_size = 50, f_type = 0, loss_type = 0):
    """
    f_type:
        0 -> triplet loss with sigmoid output
        1 -> triplet loss with binary output
        2 -> structured loss with sigmoid output
        3 -> structured loss with binary output
    loss_type:
        0 -> triplet_loss
        1 -> mean_absolute_error
        2 -> identity_loss
    """
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

    if f_type == 0 or f_type == 2:
        hash_fc = Dense(output_size, activation="sigmoid")(merged_fc)
    else:
        hash_fc = Dense(output_size,
                        activation=Lambda(lambda z: tf.divide(
                            tf.add(
                                K.sign(
                                    tf.subtract(keras.layers.activations.sigmoid(x=z), 0.5)),
                                K.abs(
                                    K.sign(
                                        tf.subtract(keras.layers.activations.sigmoid(x=z), 0.5)))),
                            2)), kernel_initializer="lecun_normal")(merged_fc)

    anchor = Input(shape=(60, 160, 3))
    positive = Input(shape=(60, 160, 3))
    negative = Input(shape=(60, 160, 3))
    p_negative = Input(shape=(60, 160, 3))

    reid_model = Model(inputs=[input_shape], outputs=[hash_fc])

    anchor_embed = reid_model(anchor)
    positive_embed = reid_model(positive)
    negative_embed = reid_model(negative)

    if f_type > 1:
        p_negative_embed = reid_model(p_negative)
        merged_output = concatenate(
            [anchor_embed, positive_embed, negative_embed, p_negative_embed])
    else:
        merged_output = concatenate([anchor_embed, positive_embed, negative_embed])

    if loss_type == 1:
        loss = Lambda(triplet_loss, (1,))(merged_output)
        if f_type <= 1:
            model = Model(inputs=[anchor, positive, negative], outputs=loss)
        else:
            #TODO: Aqui le estamos pasando esto al loss de 3 parametros
            model = Model(inputs=[anchor, positive, negative, p_negative], outputs=loss)
        model.compile(optimizer='Adam', loss='mae',
                      metrics=["mae"])
        return model

    if loss_type == 0:
        if f_type <= 1:
            model = Model(inputs=[anchor, positive, negative],
                          outputs=[merged_output])
            model.compile(optimizer='Adam', loss=multi_input_triplet_loss,
                          metrics=[multi_input_triplet_loss])
        else:
            model = Model(inputs=[anchor, positive,
                                  negative, p_negative], outputs=[merged_output])
            model.compile(optimizer='Adam', loss=structured_triplet_loss,
                          metrics=[structured_triplet_loss])
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
    #Esto de abajo son 3 arrays de numpy que representan imagenes RGB
    #Cada posicion es una imagen RGB de 60(ancho)x160(alto)
    x_anchor, x_positive, x_negative = triplets_mining.get_hard_triplets(
        no_triplets, data_type)
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
    return l, [x_anchor, x_positive, x_negative]


def validation_set(img_x, img_y, no_triplets, data_type):
    xt_anchor, xt_positive, xt_negative = triplets_mining.get_valid_validation_triplets(
        no_triplets, data_type)
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
    return lt, [xt_anchor, xt_positive, xt_negative]


def run_model(num_epochs=10, batch_size=128, img_x=60, img_y=160, training_size=10, validation_size=5, data_type=0):
    l, x = training_set(img_x, img_y, training_size, data_type)
    lt, x_test = validation_set(img_x, img_y, validation_size, data_type)
    cnn_model = build_model(img_x, img_y)
    # Print the model structure
    print(cnn_model.summary())

    history = AccuracyHistory()
    cnn_model.fit(x=x, y=np.zeros(l),
                  batch_size=batch_size,
                  epochs=num_epochs,
                  verbose=1,
                  validation_split=0.2,
                  #validation_data=(x_test, np.zeros(lt)),
                  callbacks=[history])

    score = cnn_model.evaluate(x=x_test, y=np.zeros(lt), verbose=0)
    print('Test loss:', score[0])
    #print('Test accuracy:', score[1])

    #Save wights and bias as numpy arrays
    np.save("np_output_weights_tripletloss",
            cnn_model.layers[3].get_weights())

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


def evaluate_model(dict_path, num_epochs=10, batch_size=128, img_x=60, img_y=160,
                    training_size=10, validation_size=5, data_type=0,
                    output_size=50, f_type=0, loss_type=0):
    #triplet_loss_input.generate_input_data()
    #triplet_loss_input.generate_input_data(1)

    lt, x_test = validation_set(img_x, img_y, validation_size, data_type)
    cnn_model = build_model(img_x, img_y, output_size, f_type, loss_type)

    print(cnn_model.summary())

    history = AccuracyHistory()
    for epoch in range(num_epochs):
        print('Epoch %s' % epoch)
        l, x = training_set(img_x, img_y, training_size, data_type)
        cnn_model.fit(x=x,
                y=np.zeros(l),
                batch_size=batch_size,
                epochs=1,
                verbose=1,
                validation_data=(x_test, np.zeros(lt)),
                callbacks=[history])

        f = dict_path[f_type]
        if f_type < 2:
            np.save(f,
                    cnn_model.layers[3].get_weights())
        else:
            np.save(f,
                    cnn_model.layers[4].get_weights())

        get_model_output(f, img_x, img_y, True)
    get_model_output(dict_path[f_type], img_x, img_y, data_type=1)
    #build_dict(dict_path[f_type], img_x=img_x, img_y=img_y, data_type=1)

d = {0: "triplet_loss_sigmoid_weights",
    1: "triplet_loss_binary_weights",
    2: "structured_loss_sigmoid_weights",
    3: "structured_dloss_binary_weights"}

evaluate_model(d, training_size=7000, validation_size=700)
