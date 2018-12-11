import linecache
import cv2
import loading_weights
import numpy as np

def get_images_pair(line, combinations, shuffled):
    directions = linecache.getline(combinations, line).split()
    addr1, addr2, label = directions
    addr1 = linecache.getline(shuffled, int(addr1) +1).split()[0]
    addr2 = linecache.getline(shuffled, int(addr2) + 1).split()[0]
    img1 = cv2.imread(addr1, cv2.IMREAD_COLOR)
    img1 = cv2.resize(img1, (60, 160))
    img2 = cv2.imread(addr2, cv2.IMREAD_COLOR)
    img2 = cv2.resize(img2, (60, 160))
    return np.array([img1, img2]), label

def fill_scores(lines_no, combinations, shuffled, weights):
    n = 0
    s_list = []
    for i in range(1, lines_no + 1):
        imgs, l = get_images_pair(i, combinations, shuffled)
        score = loading_weights.regular_score(weights, imgs)
        s_list.append((score, l))
        n += 1
        if n == 5000:
            for item in s_list:
                with open('cutomer_trained_scores.txt', 'a') as file:
                    file.write(str(item[0]) + " " + item[1] + "\n")
            n = 0
            s_list = []
    for item in s_list:
        with open('cutomer_trained_scores.txt', 'a') as file:
            file.write(str(item[0]) + " " + item[1] + "\n")

def get_images(addr_file, lines_no):
    images = []
    for line in range(1, lines_no+1):
        path = linecache.getline(addr_file, line).split()[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (60, 160))
        images.append(img)
    return images

def fill_all_images(lines_no, combinations, shuffled, weights, images_no, output):
    comb=[]
    images = get_images(shuffled, images_no)
    for line in range(1, lines_no +1):
        directions = linecache.getline(combinations, line).split()
        comb.append(directions)
    comb = [comb[i:i+5000] for i in range(0, len(comb), 5000)]

    labels_list = []
    images_list = []
    for l_triplets in comb:
        for triplet in l_triplets:
            labels_list.append(triplet[2])
            images_list.append(images[int(triplet[0])])
            images_list.append(images[int(triplet[1])])
        scores_list = loading_weights.set_of_scores(weights, np.array(images_list))
        for i in range(len(scores_list)):
            with open(output, 'a') as file:
                file.write(str(scores_list[i]) + " " + labels_list[i] + "\n")
        labels_list = []
        images_list = []


#979329
#fill_scores(979329, "combinations0.txt", "shuffled0.txt",
#                        "triplet_loss_sigmoid_weights")

#'validation_scores.txt'
fill_all_images(979329, "combinations0.txt", "shuffled0.txt",
                "triplet_loss_sigmoid_weights_customers_training", 99552, "combinations0_scores.txt")

fill_all_images(1119297, "combinations1.txt", "shuffled1.txt",
                "triplet_loss_sigmoid_weights_customers_training", 113664, "combinations1_scores.txt")

fill_all_images(99039, "continuous0.txt", "list0.txt",
                "triplet_loss_sigmoid_weights_customers_training", 99552, "continuous0_scores.txt")

fill_all_images(113115, "continuous1.txt", "list1.txt",
                "triplet_loss_sigmoid_weights_customers_training", 113664, "continuous1_scores.txt")
