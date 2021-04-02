import os
import numpy as np
import pandas as pd
from random import randrange
import csv
import math
import operator


def load_csv_dataset(filename):
    """Load the CSV file"""

    lines = csv.reader(open(filename, "r"))

    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]  # Convert String to Float numbers
    print(dataset)
    return dataset


def mean(numbers):
    """Returns the mean of numbers"""
    return np.mean(numbers)


def stdev(numbers):
    """Returns the std_deviation of numbers"""
    return np.std(numbers)


def sigmoid(z):
    """Returns the sigmoid number"""
    return 1.0 / (1.0 + math.exp(-z))


def cross_validation_split(dataset, n_folds):
    """Split dataset into the k folds. Returns the list of k folds"""
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    """Calculate accuracy percentage"""
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(
    dataset,
    algorithm,
    n_folds,
    k,
):
    """Evaluate an algorithm using a cross validation split"""
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(
            train_set,
            test_set,
            k,
        )
        actual = [row[-1] for row in fold]

        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


#############################
#############################
########     kNN      #######
#############################
#############################
def get_classes(training_set):
    return list(set([c[-1] for c in training_set]))


def find_neighbors(distances, k):
    return distances[0:k]


def find_response(neighbors, classes):
    votes = [0] * len(classes)

    for instance in neighbors:

        # print(instance)

        for ctr, c in enumerate(classes):
            # print(c)
            if instance[-2] == c:
                votes[ctr] += 1

    return max(enumerate(votes), key=operator.itemgetter(1))


def knn(training_set, test_set, k):
    distances = []
    dist = 0

    limit = len(training_set[0]) - 1

    # generate response classes from training data
    classes = get_classes(training_set)
    prediction = list()
    for ctrl, test_instance in enumerate(test_set):

        for row in training_set:
            for x, y in zip(row[:limit], test_instance[:limit]):
                dist += (x - y) * (x - y)

            distances.append(row + [math.sqrt(dist)])

            dist = 0

        distances.sort(key=operator.itemgetter(len(distances[0]) - 1))

        # find k nearest neighbors
        neighbors = find_neighbors(distances, k)

        # get the class with maximum votes
        index, value = find_response(neighbors, classes)

        # Display prediction
        # print(
        #     "The predicted class for sample "
        #     + str(test_instance)
        #     + " is : "
        #     + str(classes[index])
        # )
        # print("Number of votes : " + str(value) + " out of " + str(k))

        # empty the distance list
        distances.clear()

        prediction.append(classes[index])
    return prediction


#############################
#############################
######## Naive Bayes  #######
#############################
#############################


def separate_by_class(dataset):
    """Split training set by class value"""
    separated = {}
    for i in range(len(dataset)):
        row = dataset[i]
        if row[-1] not in separated:
            separated[row[-1]] = []
        separated[row[-1]].append(row)
    return separated


def model(dataset):
    """Find the mean and standard deviation of each feature in dataset"""
    models = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    models.pop()  # Remove last entry because it is class value.
    return models


def model_by_class(dataset):
    """find the mean and standard deviation of each feature in dataset by their class"""
    separated = separate_by_class(dataset)
    class_models = {}
    for (classValue, instances) in separated.items():
        class_models[classValue] = model(instances)
    return class_models


def calculate_pdf(x, mean, stdev):
    """Calculate probability using gaussian density function"""
    if stdev == 0.0:
        if x == mean:
            return 1.0
        else:
            return 0.0
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return 1 / (math.sqrt(2 * math.pi) * stdev) * exponent


def calculate_class_probabilities(models, input):
    """Calculate the class probability for input sample. Combine probability of each feature"""
    probabilities = {}
    for (classValue, classModels) in models.items():
        probabilities[classValue] = 1
        for i in range(len(classModels)):
            (mean, stdev) = classModels[i]
            x = input[i]
            probabilities[classValue] *= calculate_pdf(x, mean, stdev)
    return probabilities


def predict(models, inputVector):
    """Compare probability for each class. Return the class label which has max probability."""
    probabilities = calculate_class_probabilities(models, inputVector)
    (bestLabel, bestProb) = (None, -1)
    for (classValue, probability) in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(models, testSet):
    """Get class label for each value in test set."""
    predictions = []
    for i in range(len(testSet)):
        result = predict(models, testSet[i])
        predictions.append(result)
    return predictions


def naive_bayes(
    train,
    test,
):
    """Create a naive bayes model. Then test the model and returns the testing result."""
    summaries = model_by_class(train)
    predictions = getPredictions(summaries, test)
    return predictions


def main():
    print("[INFO] Setting directories...")
    project_dir = os.getcwd()
    fig_dir = os.path.join(
        project_dir, "manuscript", "src", "figures", "programing_project"
    )
    tbl_dir = os.path.join(
        project_dir, "manuscript", "src", "tables", "programing_project"
    )
    data_dir = os.path.join(project_dir, "Dataset", "programing_project")
    canc_dataset_file = os.path.join(data_dir, "breast-cancer-wisconsin.data")
    cars_dataset_file = os.path.join(data_dir, "car.data")
    ecol_dataset_file = os.path.join(data_dir, "ecoli.data")
    lett_dataset_file = os.path.join(data_dir, "letter-recognition.data")
    mush_dataset_file = os.path.join(data_dir, "mushroom.data")

    n_folds = 5
    k = 3
    #############################
    #############################
    ############ canc  ##########
    #############################
    #############################
    print("[INFO] loading and preparing the first dataset, (canc)...")
    names = ["feature_" + str(i) for i in range(9)] + ["target"]

    canc = (
        pd.read_csv(
            canc_dataset_file,
            header=0,
            names=names,
            index_col=0,
            parse_dates=True,
            squeeze=True,
        )
        .replace("?", np.NaN)
        .dropna()
    )

    dataset = np.array(canc.values).astype(np.float).tolist()

    print("--------- first dataset 'Cancer' --------------")
    print("------------------- kNN -----------------------")
    # accuracy_knn = evaluate_algorithm(dataset, knn, n_folds, k)
    # print("Naive Bayes Classification")
    # print("Accuracy in each fold: %s" % accuracy_knn)
    # print("Average Accuracy: %f" % (sum(accuracy_knn) / len(accuracy_knn)))
    #############################
    #############################
    ############ cars  ##########
    #############################
    #############################
    print("[INFO] loading and preparing the Second dataset, (cars)...")
    names = ["feature_" + str(i) for i in range(6)] + ["target"]

    cars = (
        pd.read_csv(
            cars_dataset_file,
            header=0,
            names=names,
            index_col=0,
            parse_dates=True,
            squeeze=True,
        ).dropna()
    ).reset_index()

    cars.replace(
        {
            "feature_0": {"vhigh": 4, "med": 2, "high": 3, "low": 1},
            "feature_1": {"vhigh": 4, "med": 2, "high": 3, "low": 1},
            "feature_2": {"2": 1, "3": 2, "4": 3, "5more": 4},
            "feature_3": {"2": 1, "4": 2, "more": 3},
            "feature_4": {"small": 1, "med": 2, "big": 3},
            "feature_5": {"med": 2, "high": 3, "low": 1},
            "target": {"good": 1, "vgood": 2, "acc": 3, "unacc": 4},
        },
        inplace=True,
    )

    dataset = np.array(cars.values).astype(np.float).tolist()

    print("--------- first dataset 'cars' --------------")
    print("------------------- kNN -----------------------")
    accuracy_knn = evaluate_algorithm(dataset, knn, n_folds, k)
    print("Naive Bayes Classification")
    print("Accuracy in each fold: %s" % accuracy_knn)
    print("Average Accuracy: %f" % (sum(accuracy_knn) / len(accuracy_knn)))

    #############################
    #############################
    ############ ecol  ##########
    #############################
    #############################
    print("[INFO] loading and preparing the third dataset, (ecol)...")
    names = ["feature_" + str(i) for i in range(8)] + ["target"]

    ecol = (
        pd.read_csv(
            ecol_dataset_file,
            header=0,
            names=names,
            index_col=0,
            parse_dates=True,
            squeeze=True,
        ).dropna()
    ).reset_index()

    # print(set(ecol["target"]))
    ecol.replace(
        {
            "target": {
                "im": 1,
                "om": 2,
                "imL": 3,
                "pp": 4,
                "imU": 5,
                "cp": 6,
                "imS": 7,
                "omL": 8,
            },
        },
        inplace=True,
    )
    ecol.drop(columns=["feature_0"], inplace=True)
    dataset = np.array(ecol.values).astype(np.float).tolist()

    print("--------- third dataset 'ecol' --------------")
    print("------------------- kNN -----------------------")
    # accuracy_knn = evaluate_algorithm(dataset, knn, n_folds, k)
    # print("Naive Bayes Classification")
    # print("Accuracy in each fold: %s" % accuracy_knn)
    # print("Average Accuracy: %f" % (sum(accuracy_knn) / len(accuracy_knn)))

    #############################
    ############ lett  ##########
    #############################
    #############################
    print("[INFO] loading and preparing the fourth dataset, (lett)...")
    names = ["target"] + ["feature_" + str(i) for i in range(16)]
    lett = (
        pd.read_csv(
            lett_dataset_file,
            header=0,
            names=names,
            index_col=0,
            parse_dates=True,
            squeeze=True,
        ).dropna()
    ).reset_index()
    # print(set(lett["target"]))
    lett.replace(
        {
            "target": {
                "Y": 1,
                "H": 2,
                "C": 3,
                "F": 4,
                "S": 5,
                "I": 6,
                "J": 7,
                "M": 8,
                "A": 9,
                "L": 10,
                "P": 11,
                "N": 12,
                "Q": 13,
                "X": 14,
                "V": 15,
                "T": 16,
                "D": 17,
                "U": 18,
                "K": 19,
                "Z": 20,
                "R": 21,
                "W": 22,
                "O": 23,
                "B": 24,
                "G": 25,
                "E": 26,
            }
        },
        inplace=True,
    )
    lett.drop(columns=["feature_0"], inplace=True)
    lett["target_1"] = lett["target"]
    lett.drop(columns=["target"], inplace=True)

    dataset = np.array(lett.values).astype(np.float).tolist()

    print("--------- fourth dataset 'lett' --------------")
    print("------------------- kNN -----------------------")
    accuracy_knn = evaluate_algorithm(dataset, knn, n_folds, k)
    print("Naive Bayes Classification")
    print("Accuracy in each fold: %s" % accuracy_knn)
    print("Average Accuracy: %f" % (sum(accuracy_knn) / len(accuracy_knn)))

    #############################
    ############ mush  ##########
    #############################
    #############################
    print("[INFO] loading and preparing the fifth dataset, (mush)...")
    names = ["target"] + ["feature_" + str(i) for i in range(22)]
    mush = (
        pd.read_csv(
            mush_dataset_file,
            header=0,
            names=names,
            index_col=0,
            parse_dates=True,
            squeeze=True,
        )
        .replace("?", np.NaN)
        .dropna()
    ).reset_index()

    # print(set(mush["feature_19"]))

    mush.replace(
        {
            "target": {"e": 1, "p": 2},
            "feature_0": {"x": 1, "c": 2, "f": 3, "k": 4, "s": 5, "b": 6},
            "feature_1": {"s": 1, "g": 2, "y": 3, "f": 4},
            "feature_2": {
                "r": 1,
                "u": 2,
                "y": 3,
                "p": 4,
                "w": 5,
                "c": 6,
                "e": 7,
                "n": 8,
                "g": 9,
                "b": 10,
            },
            "feature_3": {"f": 1, "t": 2},
            "feature_4": {
                "l": 1,
                "f": 2,
                "y": 3,
                "m": 4,
                "c": 5,
                "p": 6,
                "a": 7,
                "s": 8,
                "n": 9,
            },
            "feature_5": {"f": 1, "a": 2},
            "feature_6": {"w": 1, "c": 2},
            "feature_7": {"b": 1, "n": 2},
            "feature_8": {
                "w": 1,
                "y": 2,
                "e": 3,
                "n": 4,
                "k": 5,
                "g": 6,
                "u": 7,
                "b": 8,
                "r": 9,
                "p": 10,
                "h": 11,
                "o": 12,
            },
            "feature_9": {"e": 1, "t": 2},
            "feature_10": {"b": 1, "c": 2, "e": 3, "r": 4},  ##
            "feature_11": {"f": 1, "y": 2, "s": 3, "k": 4},
            "feature_12": {"f": 1, "y": 2, "s": 3, "k": 4},
            "feature_13": {
                "w": 1,
                "y": 2,
                "p": 3,
                "o": 4,
                "n": 5,
                "e": 6,
                "c": 7,
                "g": 8,
                "b": 9,
            },
            "feature_14": {
                "w": 1,
                "y": 2,
                "p": 3,
                "o": 4,
                "n": 5,
                "e": 6,
                "c": 7,
                "g": 8,
                "b": 9,
            },
            "feature_15": {"p": 1},
            "feature_16": {"o": 1, "w": 2, "y": 3, "n": 4},
            "feature_17": {"o": 1, "n": 2, "t": 3},
            "feature_18": {"e": 1, "f": 2, "l": 3, "p": 4, "n": 5},
            "feature_19": {
                "y": 1,
                "b": 2,
                "n": 3,
                "k": 4,
                "w": 5,
                "r": 6,
                "o": 7,
                "h": 8,
                "u": 9,
            },
            "feature_20": {
                "y": 1,
                "n": 2,
                "v": 3,
                "c": 4,
                "a": 5,
                "s": 6,
            },
            "feature_21": {
                "l": 1,
                "w": 2,
                "d": 3,
                "p": 4,
                "g": 5,
                "m": 6,
                "u": 7,
            },
        },
        inplace=True,
    )

    mush["target_1"] = mush["target"]
    mush.drop(columns=["target"], inplace=True)

    dataset = np.array(mush.values).astype(np.float).tolist()

    print("--------- fifth dataset 'mush' --------------")
    print("------------------- kNN -----------------------")
    accuracy_knn = evaluate_algorithm(dataset, knn, n_folds, k)
    print("Naive Bayes Classification")
    print("Accuracy in each fold: %s" % accuracy_knn)
    print("Average Accuracy: %f" % (sum(accuracy_knn) / len(accuracy_knn)))


if __name__ == "__main__":
    main()
