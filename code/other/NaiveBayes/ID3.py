import os
import numpy as np
import pandas as pd
from random import randrange
import csv
import math
import pprint as pp


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
        )
        actual = [row[-1] for row in fold]
        # print(actual)
        # print(type(actual))
        # print(train_set)
        # print(test_set)
        # 1 / 0
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def predict(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            # # 2.
            try:
                result = tree[key][query[key]]
            except:
                print("query.key:", query[key])
                print("key:", key)
                pp.pprint(tree)
                1 / 0
                return default

            # # 3.
            # print("key:", query[key])
            # print("tree.keys()", list(tree.keys()))
            # print(tree[key][query[key]])
            result = tree[key][query[key]]
            # 4.
            if isinstance(result, dict):
                return predict(query, result)

            else:
                return result


def test(data, tree, features_name):
    # Create new query instances by simply removing the target feature column from the original dataset and
    # convert it to a dictionary
    # Zip = zip(features_name, data)
    # Create a dictionary from zip object
    # queries = dict(Zip)

    Df = pd.DataFrame(data, columns=features_name)
    queries = Df.iloc[:, :-1].to_dict(orient="records")

    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])

    # Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 4.0)
    return predicted


# Entropy Calculation method
def calc_entropy(data):
    # Calculate the length of the data-set
    entries = len(data)
    labels = {}
    # Read the class labels from the data-set file into the dict object "labels"
    for row in data:
        label = row[-1]
        if label not in labels.keys():
            labels[label] = 0
        labels[label] += 1
    # entropy variable is initialized to zero

    entropy = 0.0
    # For every class label (x) calculate the probability p(x)
    for key in labels:
        prob = float(labels[key]) / entries
        # Entropy formula calculation
        entropy -= prob * math.log(prob, 2)
    # print "Entropy -- ",entropy
    # Return the entropy of the data-set
    return entropy


# Function to determine the best attribute for the split criteria
def attribute_selection(data):
    # get the number of features available in the given data-set
    features = len(data[0]) - 1
    # Fun call to calculate the base entropy (entropy of the entire data-set)
    baseEntropy = calc_entropy(data)
    # initialize the info-gain variable to zero
    max_InfoGain = 0.0
    bestAttr = -1
    # iterate through the features identified
    for i in range(features):
        # store the values of the features in a variable
        AttrList = [row[i] for row in data]

        # get the unique values from the feature values
        uniqueVals = set(AttrList)

        # initializing the entropy and the attribute entropy to zero
        newEntropy = 0.0
        attrEntropy = 0.0
        # iterate through the list of unique values and perform split
        for value in uniqueVals:
            # function call to split the data-set
            newData = dataset_split(data, i, value)
            # probability calculation
            prob = len(newData) / float(len(data))
            # entropy calculation for the attributes
            newEntropy = prob * calc_entropy(newData)
            attrEntropy += newEntropy
        # calculation of Information Gain
        infoGain = baseEntropy - attrEntropy
        # identify the attribute with max info-gain
        if infoGain > max_InfoGain:
            max_InfoGain = infoGain
            bestAttr = i
    # return the attribute identified
    return bestAttr


# Function to split the data-set based on the attribute that has maximum information gain
# input values: data-set, attribute index and attribute-value
def dataset_split(data, arc, val):
    # declare a list variable to store the newly split data-set
    newData = []
    # iterate through every record in the data-set and split the data-set
    for row in data:
        if row[arc] == val:
            reducedSet = list(row[:arc])
            reducedSet.extend(row[arc + 1 :])
            newData.append(reducedSet)
    # return the new list that has the data-set that is split on the selected attribute
    return newData


# Function to build the decision tree
def decision_tree(data, labels):
    # list variable to store the class-labels (terminal nodes of decision tree)
    classList = [row[-1] for row in data]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # functional call to identify the attribute for split

    maxGainNode = attribute_selection(data)

    # variable to store the class label value
    treeLabel = labels[maxGainNode]
    # dict object to represent the nodes in the decision tree
    theTree = {treeLabel: {}}
    del labels[maxGainNode]
    # get the unique values of the attribute identified
    nodeValues = [row[maxGainNode] for row in data]
    uniqueVals = set(nodeValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # update the non-terminal node values of the decision tree
        theTree[treeLabel][value] = decision_tree(
            dataset_split(data, maxGainNode, value), subLabels
        )
    # return the decision tree (dict object)
    return theTree


def id3(train_set, test_set):
    names = ["feature_" + str(i) for i in range(9)]
    tree = decision_tree(train_set, names)
    pp.pprint(tree)
    names = ["feature_" + str(i) for i in range(9)] + ["target"]
    predicted = test(test_set, tree, names)
    return predicted.values.T.squeeze()


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
    # tree = decision_tree(dataset, names)
    # names = ["feature_" + str(i) for i in range(9)] + ["target"]
    # test([[1, 1, 1, 1, 2, 10, 3, 1, 1, 2], [2, 1, 2, 1, 2, 1, 3, 1, 1, 4]], tree, names)
    # 1 / 0
    # print(dataset)
    print("--------- first dataset 'Cancer' --------------")
    print("---------- Gaussian Naive Bayes ---------------")
    accuracy_id3 = evaluate_algorithm(dataset, id3, n_folds)
    print("Naive Bayes Classification")
    print("Accuracy in each fold: %s" % accuracy_id3)
    print("Average Accuracy: %f" % (sum(accuracy_id3) / len(accuracy_id3)))


if __name__ == "__main__":
    main()
