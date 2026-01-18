#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *
from collections import defaultdict, Counter

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 1: binary classification
############################################################

############################################################
# Problem 1a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # ### START CODE HERE ###
    features = Counter(x.split())
    return features
    # ### END CODE HERE ###


############################################################
# Problem 1b: stochastic gradient descent

T = TypeVar("T")


def learnPredictor(
    trainExamples: List[Tuple[T, int]],
    validationExamples: List[Tuple[T, int]],
    featureExtractor: Callable[[T], FeatureVector],
    numEpochs: int,
    eta: float,
) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    """
    weights = defaultdict(float) 

    def hingeLoss(phi, y, w):
        return max(1 - dotProduct(w, phi) * y, 0)

    def hingeLossGradient(phi, y, w):
        return  ({k: -y * v  if hingeLoss(phi, y, w) > 0 else 0 for k, v in phi.items()})

    def predictor(x):
        return 1 if dotProduct(weights, featureExtractor(x)) >= 0 else -1

    trainFeatures = [(featureExtractor(x), y) for x, y in trainExamples]
    for epoch in range(numEpochs):
        for phi, y in trainFeatures:
            loss = hingeLossGradient(phi, y, weights)
            increment(weights, -eta, loss)
    
    # ### START CODE HERE ###
    # ### END CODE HERE ###
    return weights


############################################################
# Problem 1c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = {}
        y = None
        for feature, weight in weights.items():
            if weight:
                phi[feature] = random.randrange(1, 100)
        score = dotProduct(weights, phi)
        y = 1 if score >= 0 else -1
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


def testGenerate():
    weights = {'hello': 0.7, 'world': -0.3}
    data = generateDataset(10000, weights)
    trainData = data[:8000]
    testData = data[8000:]
    predictedWeights = learnPredictor(trainData, testData, lambda x: x, 1000, 0.001)
    print(weights, predictedWeights)
    


############################################################
# Problem 1d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x):
        ngrams = {}
        x = ''.join(x.split())
        for i in range(len(x) - n + 1):
            ngrams[x[i:i+n]] = ngrams.get(x[i:i+n], 0) + 1
        return ngrams

    return extract


############################################################
# Problem 1e:
#
# Helper function to test 1e.
#
# To run this function, run the command from termial with `n` replaced
#
# $ python -c "from submission import *; testValuesOfN(n)"
#


def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be submitted.
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
            "Official: train error = %s, validation error = %s"
            % (trainError, validationError)
        )
    )


############################################################
# Problem 2b: K-means
############################################################


def kmeans(
    examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List, List, float]:
    """
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """
    centroids = random.sample(examples, K)
    assignments = [0] * len(examples)
    squaredNormXs = [sum(value ** 2 for value in example.values()) for example in examples]
    prevLoss = float('inf')
    for _ in range(maxEpochs):
        loss = 0
        squaredNormMu = [sum(value ** 2 for value in centroid.values()) for centroid in centroids]
        for i, example in enumerate(examples):
            squaredNormX = squaredNormXs[i]
            def squaredEuclideanDistance(y):
                dot = sum(example[f] * centroids[y][f] for f in example)
                return squaredNormX - 2 * dot + squaredNormMu[y]
            dists = [squaredEuclideanDistance(j) for j in range(K)]
            assignments[i] = min(range(K), key=lambda j: dists[j])
            loss += dists[assignments[i]]
        if loss == prevLoss:
            break
        centroids = []
        for j in range(K):
            centroid = defaultdict(float)
            count = 0
            for i, example in enumerate(examples):
                if assignments[i] == j:
                    count += 1
                    for k, v in example.items():
                        centroid[k] += v
            for feature in centroid:
                centroid[feature] /= count
            centroids.append(centroid)
        prevLoss = loss
    
    return centroids, assignments, loss

                    
