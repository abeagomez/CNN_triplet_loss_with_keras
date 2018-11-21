import data_reader
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
from scipy.spatial import distance

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
    hash_fc = Dense(100, activation="sigmoid")(merged_fc)

    model = Model(inputs=[input_shape], outputs=[hash_fc])

    return model

def hamming_distance(x, y):
    dist = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            dist += 1
    return dist

def build_dict(weights_file, x_train, x_labels, img_x=60, img_y=160, save=False, data_type=1):
    r = get_model_output(weights_file, x_train, x_labels, img_x, img_y, save, data_type)
    #r = np.round(r)
    d = {}
    for i in range(len(x_labels)):
        if x_labels[i] in d:
            d[x_labels[i]].append(r[i])
        else:
            d[x_labels[i]] = [r[i]]
    return d


def true_positives_and_false_negatives(d, alpha, distance=distance.euclidean):
    t_p, f_n = 0, 0
    for k in d:
        for i in range(0,len(d[k])):
            for j in range(0, len(d[k])):
                if i != j and  distance(d[k][i],d[k][j]) <= alpha:
                    t_p += 1
                else:
                    f_n += 1
    return t_p, f_n


def false_positives_and_true_negatives(d, alpha, distance=distance.euclidean):
    f_p, t_n = 0, 0
    for k in d:
        for i in range(0,len(d[k])):
            t1, t2 = __check_all_keys__(d, k, i, alpha, distance)
            f_p += t1
            t_n += t2
    return f_p, t_n


def __check_all_keys__(d, k, index, alpha, distance):
    f_p, t_n = 0, 0
    for key in d:
        if key != k:
            for element in d[key]:
                if distance(element, d[k][index]) > alpha:
                    t_n += 1
                else:
                    f_p += 1
    return f_p, t_n

def get_model_output(weights_file, x_train, x_labels, img_x=60, img_y=160, save=False, data_type=0):
    """
    weights_file: name of the files where the weights are storaged
    data: data input for the network
    save: boolean, the output of the network will be save into a csv or not
    data_type: 0 for mining data, 1 for validation data

    return value: the output of the network
    """
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 3)
    input_shape = (img_x, img_y, 3)
    x_train = x_train.astype('float32')
    x_train /= 255

    model = build_model(img_x, img_y)

    weights = np.load(weights_file + ".npy")
    model.layers[1].set_weights([weights[0], weights[1]])
    model.layers[1].trainable = False
    model.layers[3].set_weights([weights[2], weights[3]])
    model.layers[3].trainable = False
    model.layers[5].set_weights([weights[4], weights[5]])
    model.layers[5].trainable = False
    model.layers[7].set_weights([weights[6], weights[7]])
    model.layers[7].trainable = False
    model.layers[10].set_weights([weights[8], weights[9]])
    model.layers[10].trainable = False
    model.layers[11].set_weights([weights[10], weights[11]])
    model.layers[11].trainable = False
    model.layers[13].set_weights([weights[12], weights[13]])
    model.layers[13].trainable = False

    r = model.predict(x_train,
                      batch_size=128,
                      verbose=1)

    if save:
        if data_type:
            np.savetxt("outputs_for_validation.csv", r, delimiter=",")
        else:
            np.savetxt("outputs_for_mining.csv", r, delimiter=",")

    # print((r[0]))
    # print(np.round(r[0]))
    # print(r[1])
    # print(np.round(r[1]))
    # print(r[45])
    # print(np.round(r[45]))
    return r

#### Reading model structure #### (saved here as doc only)
# for i in range(0,len(model.layers)):
#     print("Esta es la capa")
#     print(i)
#     print(model.layers[i].get_config())
#     print("")
#     print("esta capa tiene:")
#     if len(model.layers[i].get_weights()) < 1:
#         print("nada, es una capa vacia")
#         print("")
#     else:
#         print("Weights:")
#         print(len(model.layers[i].get_weights()[0]))
#         print("Biases:")
#         print(len(model.layers[i].get_weights()[1]))
#         print("")
