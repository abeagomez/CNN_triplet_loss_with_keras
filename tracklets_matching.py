from __future__ import print_function
import os
import random as rd
import numpy as np
import loading_weights
import cv2
from scipy.spatial import distance
from scipy import stats
from operator import itemgetter

path = './annotated'


def generate_cutomer_tracklets_dict():
    customers = os.listdir(path)
    c_dict = {}
    for customer in customers:
        try:
            tracklets = os.listdir(path + "/" + customer)
        except:
            continue
        else:
            c_dict[customer] = []
            for tracklet in tracklets:
                if tracklet[0] != ".":
                    c_dict[customer].append(tracklet)
    return c_dict


def generate_tracklets_images_dict(customers_dict):
    tracklets_dict = {}
    for k in customers_dict:
        for t in customers_dict[k]:
            images = os.listdir(
                path + "/" + k + "/" + t)
            tracklets_dict[t] = []
            for i in images:
                if i[-3::] == "png":
                    tracklets_dict[t].append(i)
    return tracklets_dict


def get_positive_tracklets_pairs(customers_dict, tracklets_dict, pairs_no):
    pairs_list = []
    for i in range(pairs_no):
        customer = 0
        while True:
            customer = rd.choice(list(customers_dict))
            if len(customers_dict[customer]) > 1:
                break
        trc1 = 0
        trc2 = 0
        while True:
            trc1 = customers_dict[customer][rd.randint(
                0, len(customers_dict[customer])-1)]
            trc2 = customers_dict[customer][rd.randint(
                0, len(customers_dict[customer])-1)]
            if trc1 != trc2:
                break
        pairs_list.append(
            [customer, trc1, trc2, tracklets_dict[trc1], tracklets_dict[trc2], 1])
        print([customer, trc1, trc2])
    return pairs_list


def get_negative_tracklets_pairs(customers_dict, tracklets_dict, pairs_no):
    pairs_list = []
    for i in range(pairs_no):
        customer1 = 0
        customer2 = 0
        while True:
            customer1 = rd.choice(list(customers_dict))
            customer2 = rd.choice(list(customers_dict))
            if customer1 != customer2:
                break
        trc1 = customers_dict[customer1][rd.randint(
            0, len(customers_dict[customer1])-1)]
        trc2 = customers_dict[customer2][rd.randint(
            0, len(customers_dict[customer2])-1)]
        pairs_list.append(
            [customer1, customer2, trc1, trc2, tracklets_dict[trc1], tracklets_dict[trc2], 0])
        print([customer1, customer2, trc1, trc2])
    return pairs_list


def get_images(customer1, customer2, trc1, trc2, img1, img2):
    images1 = []
    for img in img1:
        p = path + "/" + customer1 + "/" + trc1 + "/" + img
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (60, 160))
        images1.append(img)
    images2 = []
    for img in img2:
        p = path + "/" + customer2 + "/" + trc2 + "/" + img
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (60, 160))
        images2.append(img)
    return np.array(images1), np.array(images2)


def get_sample_output(pairs_no):
    customers_dict = generate_cutomer_tracklets_dict()
    tracklets_dict = generate_tracklets_images_dict(customers_dict)
    pos_data = get_positive_tracklets_pairs(
        customers_dict, tracklets_dict, pairs_no)
    neg_data = get_negative_tracklets_pairs(
        customers_dict, tracklets_dict, pairs_no)
    pairs = []
    for data in pos_data:
        images1, images2 = get_images(data[0], data[0], data[1],
                                      data[2], data[3], data[4])
        output1 = loading_weights.get_model_output(
            "triplet_loss_sigmoid_weights_customers_training", images1)
        output2 = loading_weights.get_model_output(
            "triplet_loss_sigmoid_weights_customers_training", images2)
        pairs.append([output1, output2, 1])
    for data in neg_data:
        images1, images2 = get_images(data[0], data[1], data[2],
                                      data[3], data[4], data[5])
        output1 = loading_weights.get_model_output(
            "triplet_loss_sigmoid_weights_customers_training", images1)
        output2 = loading_weights.get_model_output(
            "triplet_loss_sigmoid_weights_customers_training", images2)
        pairs.append([output1, output2, 0])
    return pairs


def central_tendency_measures(outputs_list):
    measures = []
    for output in outputs_list:
        distances = []
        for i in range(0, len(output[0])):
            for j in range(0, len(output[1])):
                distances.append(round(distance.euclidean(
                    output[0][i], output[1][j]), 4))
        mean = np.mean(distances)
        median = np.median(distances)
        mode = stats.mode(distances)
        measures.append([mean, median, mode, output[2]])
    return measures


outputs_list = get_sample_output(2500)
measures = central_tendency_measures(outputs_list)
with open("central_measures_results.txt") as file:
    for i in sorted(measures, key=itemgetter(1)):
        file.write(str(i) + "\n")
