import numpy as np
import data_reader
import random as rd

output_data = np.genfromtxt('outputs_for_mining.csv', delimiter=',')

def transform_to_binary(data):
    for output in data:
        for i in range(0,len(output)):
            if output[i] < 0.5:
                output[i] = 0
            else:
                output[i] = 1
    return data

output_data = transform_to_binary(output_data)

def get_random_triplets():
    training_data = data_reader.get_training_set()
    images, labels = training_data[0], training_data[1]
    anchor = []
    positive = []
    negative = []
    for i in range(0,100):
        anchor.append(images[rd.randint(0, len(images)-1)])
        positive.append(images[rd.randint(0, len(images)-1)])
        negative.append(images[rd.randint(0, len(images)-1)])
    return np.array(anchor), np.array(positive), np.array(negative)


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
