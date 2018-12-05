def average_precision(scores, trh):
    tp, fp= 0, 0
    precisions = []
    for i in scores:
        if i[1] < trh:
            if i[0] == 1:
                tp += 1
            else:
                fp += 1
            if tp == 0 and fp == 0:
                precisions.append(0)
            else:
                precisions.append(tp/(tp + fp))
        else:
            break
    average = sum(precisions)/float(len(precisions))
    return average

def mAP(scores, alpha_list):
    av = []
    for i in alpha_list:
        av.append(average_precision(scores, i))
    return sum(av)/float(len(av))
