import numpy as np
import data_reader
import random as rd



def transform_to_binary(data):
    for output in data:
        for i in range(0,len(output)):
            if output[i] < 0.5:
                output[i] = 0
            else:
                output[i] = 1
    return data

#Set to 50 random triplets
def get_random_triplets():
    training_data = data_reader.get_training_set()
    images, labels = training_data[0], training_data[1]
    anchor = []
    positive = []
    negative = []
    for i in range(0,50):
        anchor.append(images[rd.randint(0, len(images)-1)])
        positive.append(images[rd.randint(0, len(images)-1)])
        negative.append(images[rd.randint(0, len(images)-1)])
    return np.array(anchor), np.array(positive), np.array(negative)

def get_hard_triplets():
    anchor_set = []
    positive_set = []
    negative_set = []
    triplets = 3000
    data_dict = build_dict()
    while triplets > 0:
        triplets -= 1
        anchor_dic_key = rd.randint(0, len(data_dict)-1)
        anchor_value_len = len(data_dict[anchor_dic_key])
        anchor = data_dict[anchor_index][rd.randint(
            0, len(data_dict[anchor_index])-1)]
        positive = data_dict[anchor_index][rd.randint(
            0, len(data_dict[anchor_index])-1)]
        while True:



def build_dict():
    output_data = np.genfromtxt('outputs_for_mining.csv', delimiter=',')
    output_data = transform_to_binary(output_data)
    training_data = data_reader.get_training_set()
    images, labels = training_data[0], training_data[1]
    data_dict = {}
    for i in range(0,len(images)):
        if int(labels[i]) in data_dict:
            data_dict[int(labels[i])].append((images[i], output_data[i]))
        else:
            data_dict[int(labels[i])] = [(images[i], output_data[i])]
    return data_dict

get_hard_triplets()
# t = output_data[0]

# min_dist = 61
# index = 1
# for i in range(1, len(output_data)):
#     n_dist = np.count_nonzero(t != output_data[i])
#     if n_dist < min_dist:
#         min_dist = n_dist
#         index = i
# print(min_dist)
# print(index)
