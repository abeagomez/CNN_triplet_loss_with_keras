import data_reader
import triplets_mining
import loading_weights
import metrics_controller
from scipy.spatial import distance
import numpy as np
from operator import itemgetter

#get validation set
val_images, val_labels = data_reader.get_validation_set()

#Build dictionary with the network output for the validation set
#The dict structure is label => list of images
validation_output_dict = loading_weights.build_dict(
    "triplet_loss_sigmoid_weights", val_images, val_labels)

def to_binary(l, thr):
    for i in range(len(l)):
        if l[i] < thr:
            l[i] = 0
        else:
            l[i] = 1
    return l

def dict_to_binary(d, thr):
    for k in d:
        for i in range(len(d[k])):
            d[k][i] = to_binary(d[k][i], thr)
    return d

binary_d = dict_to_binary(validation_output_dict, 0.7)
scores = loading_weights.scores(binary_d, loading_weights.hamming_distance)
scores.sort(key=itemgetter(1))
np.savetxt("binary_scores", np.array(scores))
alphas = metrics_controller.compute_alphas(scores)
print(alphas)
precision, recall, accuracy = metrics_controller.get_validation_measures(binary_d, alphas, loading_weights.hamming_distance)
print(precision)
print("")
print(recall)
print("")
print(accuracy)
