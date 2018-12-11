import data_reader
import triplets_mining
import loading_weights
import numpy as np
from operator import itemgetter
from more_itertools import unique_everseen
from scipy.spatial import distance
import mAP

#get validation set
#val_images, val_labels = data_reader.get_validation_set()

#Build dictionary with the network output for the validation set
#The dict structure is label => list of images
#validation_output_dict = loading_weights.build_dict(
#    "triplet_loss_sigmoid_weights", val_images, val_labels)

# Generating scores and sorting them by score_value.
# After that we can compute the alphas
#scores = loading_weights.scores(validation_output_dict)
#scores.sort(key=itemgetter(1))
#np.savetxt("scores", np.array(scores))

def get_alpha(n, scores):
    c = 0
    for i in range(len(scores)-1, -1, -1):
        if scores[i][0] != 0:
            c += 1
            if c == n:
                return scores[i][1]

def compute_alphas(scores):
    alphas = []
    for i in range(len(scores)-1, 0, -1):
        if scores[i][0] != 0:
            alphas.append(scores[i][1])
            break
    pos = 0
    for i in scores:
        if i[0] == 1:
            pos +=1
    alphas.append(get_alpha(pos*5//100, scores))
    alphas.append(get_alpha(pos*10//100, scores))
    alphas.append(get_alpha(pos*15//100, scores))
    alphas.append(get_alpha(pos*20//100, scores))
    alphas.append(get_alpha(pos*25//100, scores))
    alphas.append(get_alpha(pos*30//100, scores))
    alphas.append(get_alpha(pos*35//100, scores))
    alphas.append(get_alpha(pos*40//100, scores))
    alphas.append(get_alpha(pos*45//100, scores))
    alphas.append(get_alpha(pos*50//100, scores))
    alphas.append(get_alpha(pos*55//100, scores))
    alphas.append(get_alpha(pos*50//100, scores))
    alphas.append(get_alpha(pos*60//100, scores))
    alphas.append(get_alpha(pos*65//100, scores))
    alphas.append(get_alpha(pos*70//100, scores))
    alphas.append(get_alpha(pos*75//100, scores))
    alphas.append(get_alpha(pos*80//100, scores))
    alphas.append(get_alpha(pos*83//100, scores))
    alphas.append(get_alpha(pos*85//100, scores))
    alphas.append(get_alpha(pos*87//100, scores))
    alphas.append(get_alpha(pos*90//100, scores))
    alphas.append(get_alpha(pos*91//100, scores))
    alphas.append(get_alpha(pos*92//100, scores))
    alphas.append(get_alpha(pos*93//100, scores))
    alphas.append(get_alpha(pos*94//100, scores))
    alphas.append(get_alpha(pos*95//100, scores))
    alphas.append(get_alpha(pos*96//100, scores))
    alphas.append(get_alpha(pos*97//100, scores))
    alphas.append(get_alpha(pos*98//100, scores))
    alphas.append(get_alpha(pos*99//100, scores))
    return alphas

#compute_alphas(scores)

# Value obtained using the compute_alphas function
alpha_values_48 = [0.5139309763908386, 0.3141639232635498, 0.2696551978588104,
                0.25099244713783264, 0.23562417924404144, 0.2227691411972046,
                0.21336081624031067, 0.2039041668176651, 0.19536618888378143,
                0.18710683286190033, 0.17919722199440002, 0.17105601727962494,
                0.17919722199440002, 0.16341888904571533, 0.15562736988067627,
                0.14752839505672455, 0.13915851712226868, 0.1305304914712906,
                0.12497586756944656, 0.12142099440097809, 0.11658533662557602,
                0.10978520661592484, 0.10728074610233307, 0.10511589795351028,
                0.10284426063299179, 0.09915345162153244, 0.09547270834445953,
                0.09161972254514694, 0.08717956393957138, 0.08022311329841614,
                0.07351968437433243]

# alpha_values_24 = [0.5415474772453308, 0.2987578213214874, 0.26313114166259766,
#                    0.24219895899295807, 0.22505079209804535, 0.21169224381446838,
#                    0.20053379237651825, 0.19064480066299438, 0.18229621648788452,
#                    0.17313770949840546, 0.1655581295490265, 0.15739388763904572,
#                    0.1655581295490265, 0.15042486786842346, 0.14395177364349365,
#                    0.13611237704753876, 0.12866441905498505, 0.12086212635040283,
#                    0.11614082008600235, 0.11236981302499771, 0.10940560698509216,
#                    0.10457708686590195, 0.10246941447257996, 0.10042474418878555,
#                    0.09769422560930252, 0.0948176383972168, 0.0910714715719223,
#                    0.0871584564447403, 0.08220498263835907, 0.07622260600328445,
#                    0.06916658580303192]

# alpha_values_32 = [0.4790276288986206, 0.315014511346817, 0.2785642743110657,
#                    0.2584196627140045,0.24326393008232117,0.23014791309833527,
#                    0.21928636729717255,0.20913469791412354,0.20141170918941498,
#                    0.19306766986846924,0.18531115353107452,0.1764465719461441,
#                    0.18531115353107452,0.16799472272396088,0.16095061600208282,
#                    0.15359869599342346,0.14625586569309235,0.1368730664253235,
#                    0.13116325438022614,0.12708482146263123,0.12367727607488632,
#                    0.1172047033905983,0.11433729529380798,0.11159161478281021,
#                    0.10799553990364075,0.10519144684076309,0.10070207715034485,
#                    0.09624914824962616,0.09093035757541656,0.08421322703361511,
#                    0.07605653256177902]

# alpha_values_128 = [0.536609411239624,0.31917864084243774, 0.2777097821235657,
#                     0.25572478771209717,0.23902268707752228,0.22613857686519623,
#                     0.21468956768512726,0.2046061009168625,0.19604137539863586,
#                     0.1888774335384369,0.1811078041791916,0.17352072894573212,
#                     0.1811078041791916,0.1656821370124817,0.15794463455677032,
#                     0.15017159283161163,0.1413281112909317,0.1323549747467041,
#                     0.12691131234169006,0.12295044958591461,0.11852174252271652,
#                     0.11159434169530869,0.1093238890171051,0.10720188170671463,
#                     0.1038406565785408,0.10100094974040985,0.09770993888378143,
#                     0.0938996970653534,0.09012474864721298,0.08377117663621902,
#                     0.07734707742929459]

def get_validation_measures(d, alphas, distance=distance.euclidean):
    precision, recall, accuracy = [], [], []
    for i in alphas:
        t_p, f_n = loading_weights.true_positives_and_false_negatives(d, i, distance)
        f_p, t_n = loading_weights.false_positives_and_true_negatives(d, i, distance)
        precision.append(t_p/(t_p + f_p))
        recall.append(t_p/(t_p + f_n))
        accuracy.append((t_p + t_n)/(t_p + f_n + f_p + t_n))
    return precision, recall, accuracy

#print(get_validation_measures())

#0.7343456904164845
#print(mAP.mAP(scores, alpha_values_48[-16:]))

#0.7010761673759461
#print(mAP.mAP(scores, alpha_values_24[-16:]))

#0.6437642889847354
#print(mAP.mAP(scores, alpha_values_32[-16:]))

#0.6727484759389591
#print(mAP.mAP(scores, alpha_values_128[-16:]))

def get_cmc_values(d):
    ranking = loading_weights.ranks_list(d)
    ranking.sort()
    total = len(ranking)
    sumarized = list(unique_everseen(ranking))
    percents = []
    acum = 0
    for i in sumarized:
        acum += ranking.count(i)
        percents.append(acum/total)
    return sumarized[:50], percents[:50]


#print(get_cmc_values(validation_output_dict))
