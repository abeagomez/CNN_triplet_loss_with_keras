import keras.backend as K
import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, merge, Input, Embedding
from keras.models import Sequential
import triplets_mining


def distance(x, y):
    return K.sqrt(K.sum(K.square(x - y), axis=-1))


def triplets_max(dist_anchor_positive, dist_anchor_negative, margin):
    x = (dist_anchor_positive - dist_anchor_negative + margin)
    return tf.maximum(x, 0)


def bpr_triplet_loss(X):
    anchor, positive, negative = X
    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(anchor * positive, axis=-1, keepdims=True) -
        K.sum(anchor * negative, axis=-1, keepdims=True))
    return loss


def build_model(img_x, img_y):
    input_shape = (img_x, img_y, 3)
    #Este es mi modelo#########################################################
    #Se que tiene dos Conv2D seguidas, es lo que decia el paper...#############
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(4024, activation='relu'))
    model.add(Dense(512, activation='sigmoid'))
    ###########################################################################

    positive_item_input = Input(
        shape=(128, 254, 3), name='positive_item_input')
    negative_item_input = Input(
        shape=(128, 254, 3), name='negative_item_input')

    # Shared embedding layer for positive and negative items
    item_embedding_layer = Embedding(
                                254, 3,
                                name='item_embedding',
                                input_length=128)

    anchor_input = Input(shape=(128, 254, 3), name = 'anchor_input')
    anchor_embedding = Embedding(254, 3,
                                 name='anchor_embedding',
                                 input_length=128)
    positive_item_embedding = Flatten()(item_embedding_layer(positive_item_input))
    negative_item_embedding = Flatten()(item_embedding_layer(negative_item_input))
    anchor_embedding = Flatten()(anchor_embedding(anchor_input))

    #loss = triplets_max(dist_anchor_positive, dist_anchor_negative, 0.05)
    loss = merge(
        [anchor_embedding, positive_item_embedding, negative_item_embedding],
        mode=bpr_triplet_loss,
        name='loss')

    model = Model(inputs=[anchor_input, positive_item_input, negative_item_input],
                    outputs=loss)
    model.compile(optimizer='Adam', loss='mean_absolute_error')
    return model


num_epochs = 10
img_x, img_y = 128, 254
#Esto de abajo son 3 arrays de numpy que representan imagenes RGB
#Cada posicion es una imagen RGB de 128(ancho)x254(alto)
x_anchor, x_positive, x_negative = triplets_mining.get_random_triplets()

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

for epoch in range(num_epochs):
    print('Epoch %s' % epoch)

    #X = {
    #    'anchor': x_anchor,
    #    'positive': x_positive,
    #    'negative': x_negative,
    #}

    model.fit([x_anchor, x_positive, x_negative],
              batch_size=64,
              nb_epoch=1,
              verbose=1,
              shuffle=True)
