import loading_weights
import numpy as np
from scipy.spatial import distance

#r = loading_weights.get_model_output("testing_weights", data_type=1)
#print(r[0])

d = loading_weights.build_dict("testing_weights")
alpha = 2
t_p, f_n = loading_weights.true_positives_and_false_negatives(d, alpha)
f_p, t_n = loading_weights.false_positives_and_true_negatives(d, alpha)

print("Con distancia euclidiana y sin redondear")
print("Alpha = %.3f y Total:" %alpha)
print(t_p + f_n + f_p + t_n)
print("True Positives = %d" %t_p)
print("True negatives = %d" %t_n)
print("False Positives = %d" %f_p)
print("False Negatives = %d" %f_n)
print("")
print("Precision = %.5f" %(t_p/(t_p + f_p)))
print("Recall = %.5f" %(t_p/(t_p + f_n)))
print("Accuracy = %.5f" % ((t_p + t_n)/(t_p + f_n + f_p + t_n)))