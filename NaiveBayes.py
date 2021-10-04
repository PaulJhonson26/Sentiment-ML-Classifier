import numpy

def main():


    DtrainFile = open("train_dataset.tsv", "r")
    Dtrain = numpy.loadtxt(DtrainFile, delimiter="\t")

#    print(Dtrain)
    features = Dtrain[:,:-1]
    targets = Dtrain[:,-1]
    prior_one = getPrior1(targets)
    prior_zero = 1 - prior_one

    negative_feature_likelihoods = getFeatureLikelihood(features, 0, targets)
    positive_feature_likelihoods = getFeatureLikelihood(features, 1, targets)

    print("negative", negative_feature_likelihoods)
    print("positive", positive_feature_likelihoods)

    cp = open("class_priors.tsv", "w+")
    cp.write(str(prior_one) + "\n")
    cp.write(str(prior_zero) + "\n")

    nf = open("negative_feature_likelihoods.tsv", "w+")
    for i in negative_feature_likelihoods:
        nf.write(str(i) + "\n")

    pf = open("positive_feature_likelihoods.tsv", "w+")
    for i in positive_feature_likelihoods:
        pf.write(str(i) + "\n")



def getPrior1(targetList):
    zeroCount = 0
    oneCount = 0
    for target in targetList:
        if(target == 0):
            zeroCount += 1
        elif(target == 1):
            oneCount += 1

    return(oneCount / len(targetList))

def getFeatureLikelihood(pointsList, targetValue, targets):
    likelihoodArray = []
    for var in range(1000):
        likelihood = 0
        for i, featureVector in enumerate(pointsList): #25000 featureVectors, 1000 variables in each
            if(targets[i] == targetValue):
                likelihood = likelihood + featureVector[var]
        likelihood = likelihood / 25000
        likelihoodArray.insert(var, likelihood)

    return likelihoodArray

if __name__ == "__main__":
    main()