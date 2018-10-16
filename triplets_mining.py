import numpy as np
import data_reader
import random as rd

def hamming_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.count_nonzero(a!=b)

def triplet_loss(a, p, n, data_type):
    if data_type:
        return max(hamming_distance(a,p)- hamming_distance(a,n), 0)
    else:
        return max(np.linalg.norm(a-p) - np.linalg.norm(a-n), 0)

def transform_to_binary(data):
    return [[0 if i < 0.5 else 1 for i in output] for output in data]

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

def get_valid_validation_triplets(triplets, data_type):
    print("======= Generating Validation Triplets =======")
    anchor_set = []
    positive_set = []
    negative_set = []
    data_dict = build_validation_dict()
    while triplets > 0:
        anchor_dic_key = rd.randint(0, len(data_dict)-1)
        anchor_value_len = len(data_dict[anchor_dic_key])
        anchor_value = data_dict[anchor_dic_key][rd.randint(
            0, anchor_value_len - 1)]
        anchor = anchor_value[data_type]
        anchor_image = anchor_value[2]
        positive_value = data_dict[anchor_dic_key][rd.randint(
            0, anchor_value_len - 1)]
        positive = positive_value[data_type]
        positive_image = positive_value[2]
        for k in data_dict:
            if k != anchor_dic_key:
                negative_len = len(data_dict[k])
                negative_value = data_dict[k][rd.randint(
                    0, negative_len - 1)]
                negative_image = negative_value[2]
                anchor_set.append(anchor_image)
                positive_set.append(positive_image)
                negative_set.append(negative_image)
                triplets -= 1
                break
    print("======= Validation Triplets Generation Finished =======")
    print("======= {} triplets generated =======".format(len(anchor_set)))
    return np.array(anchor_set), np.array(positive_set), np.array(negative_set)


def get_hard_value(k, anchor_dic_key, data_dict, anchor, positive, data_type):
    if k != anchor_dic_key:
        for value in data_dict[k]:
            loss = triplet_loss(
                anchor, positive, value[data_type], data_type)
            if loss != 0:
                return loss, value
    return 0, 0


#Triplets: the number of triplets we want to build
#data_type: 0 if original output, 1 if binary output
def get_hard_triplets(triplets, data_type):
    print("======= Generating Training Triplets =======")
    anchor_set = []
    positive_set = []
    negative_set = []
    data_dict = build_dict()
    while triplets > 0:
        anchor_dic_key = rd.randint(0, len(data_dict)-1)
        anchor_value_len = len(data_dict[anchor_dic_key])
        anchor_value = data_dict[anchor_dic_key][rd.randint(
            0, anchor_value_len -1)]
        anchor = anchor_value[data_type]
        anchor_image = anchor_value[2]
        positive_value = data_dict[anchor_dic_key][rd.randint(
            0, anchor_value_len -1)]
        positive = positive_value[data_type]
        positive_image = positive_value[2]
        for k in data_dict:
            loss = get_hard_value(k, anchor_dic_key, data_dict,
                            anchor, positive, data_type)
            if loss[0] != 0:
                anchor_set.append(anchor_image)
                positive_set.append(positive_image)
                negative_set.append(loss[1][2])
                triplets -= 1
                break
    print("======= Training Triplets Generation Finished =======")
    print("======= {} triplets generated =======".format(len(anchor_set)))
    return np.array(anchor_set), np.array(positive_set), np.array(negative_set)

def build_dict():
    output_data = np.genfromtxt('outputs_for_mining.csv', delimiter=',')
    binary_output_data = transform_to_binary(output_data)
    training_data = data_reader.get_training_set()
    images, labels = training_data[0], training_data[1]
    data_dict = {}
    for i in range(0,len(images)):
        if int(labels[i]) in data_dict:
            data_dict[int(labels[i])].append((output_data[i], binary_output_data[i], images[i]))
        else:
            data_dict[int(labels[i])] = [
                (output_data[i], binary_output_data[i], images[i])]
    return data_dict

def build_validation_dict():
    output_data = np.genfromtxt('outputs_for_validation.csv', delimiter=',')
    binary_output_data = transform_to_binary(output_data)
    validation_data = data_reader.get_validation_set()
    images, labels = validation_data[0], validation_data[1]
    data_dict = {}
    for i in range(0, len(images)):
        if int(labels[i]) in data_dict:
            data_dict[int(labels[i])].append(
                (output_data[i], binary_output_data[i], images[i]))
        else:
            data_dict[int(labels[i])] = [
                (output_data[i], binary_output_data[i], images[i])]
    return data_dict

#print(len(get_valid_validation_triplets(10,0)[0]))
#print(len(get_hard_triplets(300, 0)[0]))
#get_hard_triplets(10,0)
#build_dict()

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
