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
    for i in range(1, lines_no + 1):
        imgs, l = get_images_pair(i, combinations, shuffled)
        score = loading_weights.regular_score(weights, imgs)
        with open('cutomer_scores.txt', 'a') as file:
            file.write(str(score) + " " + l + "\n")


#979329
fill_scores(979329, "combinations0.txt", "shuffled0.txt",
                        "triplet_loss_sigmoid_weights")
