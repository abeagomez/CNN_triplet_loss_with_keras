import keras.backend as K
import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, merge, concatenate
from keras.models import Sequential
import triplets_mining
from keras.layers import Lambda
import numpy as np


def distance(x, y):
    return K.sqrt(K.sum(K.square(x - y), axis=-1))

def triplets_max(dist_anchor_positive, dist_anchor_negative, margin):
    x = (dist_anchor_positive - dist_anchor_negative + margin)
    return tf.maximum(x, 0)


def triplet_loss(x,y):
    anchor, positive, negative = tf.split(y,3)

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), 0.05)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss

def build_model(img_x, img_y):
    #input_shape = (img_x, img_y, 3)
    input_shape = Input(shape=(img_x, img_y, 3))

    #Modelo secuencial#########################################################
    # reid_model = Sequential()
    # reid_model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
    #              activation='relu',
    #              input_shape=input_shape))
    # reid_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # reid_model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
    # reid_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # reid_model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
    # reid_model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu'))
    # reid_model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
    # reid_model.add(Flatten())
    # reid_model.add(Dense(4024, activation='relu'))
    # reid_model.add(Dense(512, activation='sigmoid'))
    ###########################################################################
    c0 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu') (input_shape)
    m0 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) (c0)
    f = Flatten()(m0)
    d1 = Dense(4024, activation='relu')(f)
    d2 = Dense(512, activation='sigmoid')(d1)

    anchor = Input(shape=(128, 254, 3))
    positive = Input(shape=(128, 254, 3))
    negative = Input(shape=(128, 254, 3))

    reid_model = Model(inputs=[input_shape], outputs=[d2])

    anchor_embed = reid_model(anchor)
    positive_embed = reid_model(positive)
    negative_embed = reid_model(negative)
    print(anchor_embed)
    #dist_anchor_positive = distance(anchor_embed, positive_embed)
    #dist_anchor_negative = distance(anchor_embed, negative_embed)

    #loss = triplets_max(dist_anchor_positive, dist_anchor_negative, 0.05)

    #loss = merge([anchor_embed, positive_embed, negative_embed],
    #             mode=triplet_loss, output_shape=(1,))

    merged_output = concatenate([anchor_embed, positive_embed, negative_embed])
    model = Model(inputs=[anchor, positive, negative], outputs=merged_output)
    model.compile(optimizer='Adam', loss=triplet_loss, metrics=[triplet_loss])

    #model = Model(inputs=[anchor, positive, negative], outputs=loss)
    #model.compile(optimizer='Adam', loss='mean_absolute_error')
    return model


num_epochs = 10
img_x, img_y = 128, 254
#Esto de abajo son 3 arrays de numpy que representan imagenes RGB
#Cada posicion es una imagen RGB de 128(ancho)x254(alto)
x_anchor, x_positive, x_negative = triplets_mining.get_random_triplets()
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

model = build_model(img_x, img_y)
# Print the model structure
print(model.summary())
model.fit(x=[x_anchor, x_positive, x_negative],y = np.ones(l),
          batch_size=64,
          epochs=10,
          verbose=1)
# for epoch in range(num_epochs):
#     print('Epoch %s' % epoch)

#     model.fit([x_anchor, x_positive, x_negative],
#             batch_size=64,
#             epochs=1,
#             verbose=1)
