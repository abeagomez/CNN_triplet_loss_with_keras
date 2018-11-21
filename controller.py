import data_reader
import triplets_mining
import CNN_triplet_loss_functional
import loading_weights
import numpy as np

#get training data
training_data = data_reader.get_training_set()

#increase the data by doing transformations as described in the paper
images, labels = data_reader.increase_data(training_data[0], training_data[1])

#get random triplets for the first iteration of mining
triplets = triplets_mining.get_random_triplets(7000, images, labels)

#Get triplets ready as input for the network
# l is the number of elements in each triplet
l, triplets =triplets_mining.prepare_triplets(60, 160, triplets)

#Build the model using triplet loss and sigmoid activation
model = CNN_triplet_loss_functional.build_model(60, 160)
print(model.summary())

num_epochs = 10
for epoch in range(num_epochs):
    print('Epoch %s' % epoch)
    model.fit(triplets,
            y=np.zeros(l),
            batch_size=120,
            epochs=1,
            verbose=1)

    np.save("triplet_loss_sigmoid_weights",
            model.layers[3].get_weights())

val_images, val_labels = data_reader.get_validation_set()
validation_output_dict = loading_weights.build_dict(
    "triplet_loss_sigmoid_weights", val_images, val_labels)

alpha = 0.05
t_p, f_n = loading_weights.true_positives_and_false_negatives(validation_output_dict, alpha)
f_p, t_n = loading_weights.false_positives_and_true_negatives(validation_output_dict, alpha)

print("Con distancia euclidiana y sin redondear")
print("Alpha = %.3f y Total:" % alpha)
print(t_p + f_n + f_p + t_n)
print("True Positives = %d" % t_p)
print("True negatives = %d" % t_n)
print("False Positives = %d" % f_p)
print("False Negatives = %d" % f_n)
print("")
print("Precision = %.5f" % (t_p/(t_p + f_p)))
print("Recall = %.5f" % (t_p/(t_p + f_n)))
print("Accuracy = %.5f" % ((t_p + t_n)/(t_p + f_n + f_p + t_n)))
