import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import random


def sigmoidFunction(weightVector, feature):
    sigsum = 0
    sigsum = np.vdot(np.transpose(weightVector), feature)
    if sigsum >= 0:
        sigFinal = math.exp(-sigsum)
        return 1 / (1 + sigFinal)
    else:
        sigFinal = math.exp(sigsum)
        return sigFinal / (1 + sigFinal)


def fit(features, targets, weights):
   # weights.insert(1000, 1)
    max_iters = 5000
    print(weights)
    for k in range(max_iters):
        vectorSum = [0] * 1001
        for i, feature in enumerate(features):
           vectorSum += feature * (targets[i] - sigmoidFunction(weights, feature))
        newWeights = weights + 0.01* vectorSum
        if(np.linalg.norm(newWeights-weights) <= 0.0005):
            print("fuck")
            break
        weights = newWeights
        if(k% 100 == 0):
            print(k)



    return weights


def main():
    DtrainFile = open("train_dataset.tsv", "r")
    Dtrain = np.loadtxt(DtrainFile, delimiter="\t")

    features = Dtrain[:, :-1]
    #    print(Dtrain)
    bias = np.ones(features.shape[0]).T
    bias = bias.reshape(len(bias), 1)
    features = np.hstack((bias, features))
    targets = Dtrain[:, -1]
    weights = []
    print(features[0])
    print(targets[0])

    ###### set random weights
    for i in range(1001):
        weights.insert(i, 0)

    print(len(weights))
    print(weights)
    print("features: ", features)

    tolerance = 0.0005



    lr = 0.01

    finalWeights = fit(features, targets, weights)
    print(finalWeights)
    print("DONE MF!")
    nf = open("weights.tsv", "w+")
    for i in finalWeights:
        nf.write(str(i) + "\n")
    # I don't think I really get when you're supposed to add the bias. Does it ever change? or is it always 1?
    # because you can't run gradient descent if the weights are 1001 and the features 1000 right? there's a dimension problem
    # so how does the bias come into this?



if __name__ == "__main__":
    main()



