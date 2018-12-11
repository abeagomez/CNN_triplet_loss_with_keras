import linecache
import cv2
import loading_weights
import numpy as np
import triplets_mining
import CNN_triplet_loss_functional

def get_images_triplets(line, combinations, train):
    directions = linecache.getline(combinations, line).split()
    anchor_addr, pos_addr, neg_addr, cam = directions
    anchor = linecache.getline(train, int(anchor_addr) + 1).split()[0]
    positive = linecache.getline(train, int(pos_addr) + 1).split()[0]
    negative = linecache.getline(train, int(neg_addr) + 1).split()[0]
    anchor = cv2.imread(anchor, cv2.IMREAD_COLOR)
    anchor = cv2.resize(anchor, (60, 160))
    positive = cv2.imread(positive, cv2.IMREAD_COLOR)
    positive = cv2.resize(positive, (60, 160))
    negative = cv2.imread(negative, cv2.IMREAD_COLOR)
    negative = cv2.resize(negative, (60, 160))
    return anchor, positive, negative

def get_triplets(lines_no, combinations, train):
    anchor, pos, neg = [], [], []
    for i in range(1, lines_no+1):
        triplet = get_images_triplets(i, combinations, train)
        anchor.append(triplet[0])
        pos.append(triplet[1])
        neg.append(triplet[2])
    return np.array(anchor), np.array(pos), np.array(neg)


def get_cnn_input(lines_no, combinations, train):
    triplets = get_triplets(lines_no, combinations, train)
    return triplets_mining.prepare_triplets(60, 160, triplets)


#199k
l, triplets = get_cnn_input(50000, "train_combination0.txt", "train0.txt")

#Build the model using triplet loss and sigmoid activation
model = CNN_triplet_loss_functional.build_model(60, 160)
print(model.summary())

history = CNN_triplet_loss_functional.AccuracyHistory()
num_epochs = 15
alpha = 0.05
for epoch in range(num_epochs):
    print('Epoch %s' % epoch)
    model.fit(triplets,
              y=np.zeros(l),
              batch_size=250,
              epochs=1,
              verbose=1,
              callbacks=[history])

    np.save("triplet_loss_sigmoid_weights_customers_training",
            model.layers[3].get_weights())
