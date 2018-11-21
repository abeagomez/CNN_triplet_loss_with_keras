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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def structured_triplet_loss(x, y):
    xi, yi, yk, yl = tf.split(y, 4, axis=1)

    # pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # neg_dist_anch = tf.reduce_sum(tf.square(tf.subtract(anchor, anchor_neg)), 1)
    # neg_dist_pos = tf.reduce_sum(tf.square(tf.subtract(anchor, positive_neg)), 1)

    # term_anchor = tf.maximum(0.0, tf.subtract(1.0, neg_dist_anch))
    # term_positive = tf.maximum(0.0, tf.subtract(1.0, neg_dist_pos))
    # inner_max = tf.maximum(term_anchor, term_positive)
    # basic_loss = tf.maximum(tf.add(inner_max, pos_dist), 0.0)
    # loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    norm_xi_yk = tf.reduce_sum(tf.square(tf.subtract(xi, yk)), 1)
    norm_yi_yl = tf.reduce_sum(tf.square(tf.subtract(yi, yl)), 1)
    norm_xi_yi = tf.reduce_sum(tf.square(tf.subtract(xi, yi)), 1)

    dif_one_and_norm_xi_yk = tf.subtract(tf.ones(
        tf.shape(norm_xi_yk), dtype=tf.float32), norm_xi_yk)
    dif_one_and_norm_yi_yl = tf.subtract(tf.ones(
        tf.shape(norm_yi_yl), dtype=tf.float32), norm_yi_yl)

    term1 = dif_one_and_norm_xi_yk
    term2 = dif_one_and_norm_yi_yl

    max_zero_and_term1 = tf.maximum(
        tf.zeros(tf.shape(term1), dtype=tf.float32), term1)
    max_zero_and_term2 = tf.maximum(
        tf.zeros(tf.shape(term2), dtype=tf.float32), term2)

    term3 = max_zero_and_term1
    term4 = max_zero_and_term2
    max_term3_and_term4 = tf.maximum(term3, term4)

    F_xi_yi = tf.add(max_term3_and_term4, norm_xi_yi)
    max_zero_and_F_xi_yi = tf.maximum(
        tf.zeros(tf.shape(F_xi_yi), dtype=tf.float32), F_xi_yi)
    term5 = max_zero_and_F_xi_yi
    loss = tf.divide(tf.reduce_sum(term5), tf.to_float(tf.size(term5)))
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
    #loss = Lambda(structured_triplet_loss, (1,))(merged_output)

    #model = Model(inputs=[anchor, positive, a_negative, p_negative], outputs=loss)
    #model.compile(optimizer='Adam', loss='mse',
    #              metrics=["mae"])
    model = Model(inputs=[anchor, positive, a_negative,
                          p_negative], outputs=[merged_output])
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

def run_model(num_epochs=10, batch_size=128, img_x=60, img_y=160, training_size=10, validation_size=5, data_type = 0):
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

    score = cnn_model.evaluate(x=x_test, y = np.zeros(lt), verbose=0)
    print('Test loss:', score[0])
    #print('Test accuracy:', score[1])

    #Save wights and bias as numpy arrays
    np.save("np_output_weights_strloss",
            cnn_model.layers[4].get_weights())

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

run_model(training_size=1000, validation_size=10)
