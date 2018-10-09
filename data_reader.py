import cv2
import numpy as np
import os

def get_num_id(path, set):
    files = os.listdir('%s/labeled/%s' % (path, set))
    files.sort()
    return files

def sorted_file_names(path, set):
    files = os.listdir('%s/labeled/%s' % (path, set))
    files.sort()
    return files

#returns a tuple k where:
#k[0] is a list containing all the images names
#k[1] is a list with len(k[1]) == len(k[1]) containing all the images labels
def training_data(path):
    files = os.listdir('%s/labeled/%s' % (path, "train"))
    files.sort()
    return files, [i[0:4] for i in files]

def training_labels(path):
    files = os.listdir('%s/labeled/%s' % (path, "train"))
    files.sort()
    return [i[0:4] for i in files]

def read_data(path, img_names, labels):
    images = []
    for file in img_names:
        image = cv2.imread(path + file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return(np.array(images), np.array(labels))

def get_training_set():
    images_file = training_data(".")
    return read_data("./labeled/train/", images_file[0], images_file[1])

def get_training_labels():
    return training_labels(".")
