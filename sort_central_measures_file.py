from operator import itemgetter

k = []

with open("central_measures_results.txt") as f:
    l = f.readlines()
    for i in l:
        k.append([int(i.split(",")[4][1:2]), float(i.split(",")[1][1::])])


def confusion_matrix(results_list, pos_counter, goal):
    k = results_list
    index = 0
    median = 0
    for i in range(len(k)):
        if k[i][0] == 1:
            index += 1
        if index == goal:
            median = k[i][1]
            index = i + 1
            break
    t_p = goal
    f_p = index - goal
    t_n = len(results_list) - index - (pos_counter - goal)
    f_n = pos_counter - goal
    precision = t_p/(t_p + f_p)
    recall = t_p/(t_p + f_n)
    accuracy = (t_p + t_n)/(pos_counter*2)
    # true pos, false pos, true neg, false neg, precision, recall, accuracy
    return [median, t_p, f_p, t_n, f_n, precision, recall, accuracy]


ones_counter = 1000
goal = 1000
k = sorted(k, key=itemgetter(1))
while(goal > 500):
    print(confusion_matrix(k, ones_counter, goal))
    goal -= 50
