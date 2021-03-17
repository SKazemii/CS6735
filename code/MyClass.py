import os
import numpy as np
import pandas as pd
from random import randrange
import csv
import math
import operator

class classifiers:
    
class myclass:
    def __init__(
        self,
        n_folds,
        algorithm,
        dataset,
        k_neighbor=1,
        verbose=True,
        max_depth=3,
        min_size=1,
    ):
        self.n_folds = n_folds
        self.k_neighbor = k_neighbor
        self.dataset = dataset
        self.verbose = verbose
        self.max_depth = max_depth
        self.min_size = min_size

        if algorithm == "NB":
            self.algorithm = self.naivebayes
        elif algorithm == "kNN":
            self.algorithm = self.knn
        elif algorithm == "ID3":
            self.algorithm = self.id3

    def cv_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_duplicated = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_duplicated))
                fold.append(dataset_duplicated.pop(index))
            dataset_split.append(fold)
        return dataset_split

    def accuracy(self, actual_output, predicted_output):
        correct = 0
        for i in range(len(actual_output)):
            if actual_output[i] == predicted_output[i]:
                correct = correct + 1
        return correct / float(len(actual_output)) * 100.0

    def evaluate_kfold(self):
        folds = self.cv_split(self.dataset, self.n_folds)
        scores = list()
        for fold in folds:
            trainset = list(folds)
            trainset.remove(fold)
            trainset = sum(trainset, [])
            testset = list()
            for row in fold:
                row_copy = list(row)
                testset.append(row_copy)
                row_copy[-1] = None
            predicted = self.algorithm(
                trainset,
                testset,
            )
            actual_output = list()
            for row in fold:
                actual_output.append(row[-1])
            accuracy = self.accuracy(actual_output, predicted)
            scores.append(accuracy)
        return scores

    def naivebayes(self, trainset, testset):
        model = self.model_classes(trainset)
        predictions = self.getPredictions(model, testset)
        return predictions

    def getPredictions(self, models, testSet):  # change
        predictions = []
        for i in range(len(testSet)):
            result = self.predict(models, testSet[i])
            predictions.append(result)
        return predictions

    def predict(self, models, test_row):  # change
        probabilities = self.calculate_class_probabilities(models, test_row)
        (bestLabel, bestProb) = (None, -1)
        for (classValue, probability) in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def separate_classes(self, dataset):
        separatedbyclasses = {}
        for i in range(len(dataset)):
            row = dataset[i]
            if row[-1] not in separatedbyclasses:
                separatedbyclasses[row[-1]] = []
            separatedbyclasses[row[-1]].append(row)
        return separatedbyclasses

    def model_feature(self, dataset):
        """Find the mean and standard deviation of each feature in dataset"""

        models = list()
        for feature in zip(*dataset):
            models.append((np.mean(feature), np.std(feature)))

        models = models[:-1]
        return models

    def model_classes(self, dataset):
        """find the mean and standard deviation of each feature in dataset by their class"""
        separatedbyclasses = self.separate_classes(dataset)
        classmodels = {}

        for (classValue, instances) in separatedbyclasses.items():
            classmodels[classValue] = self.model_feature(instances)
        return classmodels

    def calculate_pdf(self, x, mean, stdev):  # change
        """Calculate probability using gaussian density function"""
        if stdev == 0.0:
            if x == mean:
                return 1.0
            else:
                return 0.0
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return 1 / (math.sqrt(2 * math.pi) * stdev) * exponent

    def calculate_class_probabilities(self, models, input):  # change
        """Calculate the class probability for input sample. Combine probability of each feature"""
        probabilities = {}
        for (classValue, classModels) in models.items():
            probabilities[classValue] = 1
            for i in range(len(classModels)):
                (mean, stdev) = classModels[i]
                x = input[i]
                probabilities[classValue] *= self.calculate_pdf(x, mean, stdev)
        return probabilities

    def find_response(self, neighbors, classes):  # change
        votes = [0] * len(classes)

        for instance in neighbors:

            # print(instance)

            for ctr, c in enumerate(classes):
                # print(c)
                if instance[-2] == c:
                    votes[ctr] += 1

        return max(enumerate(votes), key=operator.itemgetter(1))

    def knn(self, training_set, test_set):  # change
        distances = []
        features_number = len(training_set[0]) - 1

        # generate response classes from training data
        classes = list(set([row[-1] for row in training_set]))
        prediction = list()
        for crtl, test_instance in enumerate(test_set):

            for row in training_set:
                temp = math.dist(row[:features_number], test_instance[:features_number])

                distances.append(row + [temp])

            distances.sort(key=operator.itemgetter(len(distances[0]) - 1))

            neighbors = distances[0 : self.k_neighbor]

            # get the class with maximum votes
            index, value = self.find_response(neighbors, classes)

            # Display prediction
            if self.verbose:
                print(
                    "The predicted class for sample "
                    + str(test_instance[:-1])
                    + " is : "
                    + str(classes[index])
                )
                print(
                    "Number of votes : "
                    + str(value)
                    + " out of "
                    + str(self.k_neighbor)
                )

            distances.clear()

            prediction.append(classes[index])
        return prediction

    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Calculate the Entropy for a split dataset
    def entropy(self, groups, classes, b_score):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        entropy = 0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                if p > 0:
                    score = p * math.log(p, 2)
            # weight the group score by its relative size i.e Entrpy gain
            entropy -= score * (size / n_instances)
        return entropy

    # Select the best split point for a dataset
    def get_split(self, dataset):
        # if split_parameter == "entropy":  # this is invoked for parameter entropy
        classes = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 1, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                entropy = self.entropy(groups, classes, b_score)
                if entropy < b_score:
                    b_index, b_value, b_score, b_groups = (
                        index,
                        row[index],
                        entropy,
                        groups,
                    )
        return {"index": b_index, "value": b_value, "groups": b_groups}

    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def split(self, node, depth):
        left, right = node["groups"]
        del node["groups"]
        # check for a no split
        if not left or not right:
            node["left"] = node["right"] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= self.max_depth:
            node["left"], node["right"] = self.to_terminal(left), self.to_terminal(
                right
            )
            return
        # process left child
        if len(left) <= self.min_size:
            node["left"] = self.to_terminal(left)
        else:
            node["left"] = self.get_split(left)
            self.split(node["left"], depth + 1)
        # process right child
        if len(right) <= self.min_size:
            node["right"] = self.to_terminal(right)
        else:
            node["right"] = self.get_split(right)
            self.split(node["right"], depth + 1)

    # Build a decision tree
    def build_tree(self, train):
        root = self.get_split(train)
        self.split(root, 1)
        return root

    # Print a decision tree
    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print(
                "%s[ATTRIBUTE[%s] = %.50s]"
                % ((depth * "\t", (node["index"] + 1), node["value"]))
            )
            self.print_tree(node["left"], depth + 1)
            self.print_tree(node["right"], depth + 1)
        else:
            print("%s[%s]" % ((depth * " ", node)))

    # Make a prediction with a decision tree
    def predict_id3(self, node, row):
        if row[node["index"]] < node["value"]:
            if isinstance(node["left"], dict):
                return self.predict_id3(node["left"], row)
            else:
                return node["left"]
        else:
            if isinstance(node["right"], dict):
                return self.predict_id3(node["right"], row)
            else:
                return node["right"]

    # Classification and Regression Tree Algorithm
    def id3(self, train, test):
        tree = self.build_tree(train)
        predictions = list()
        for row in test:
            prediction = self.predict_id3(tree, row)
            predictions.append(prediction)
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
    # print(canc.head())

    dataset = np.array(canc.values).astype(np.float).tolist()
    nb = myclass(n_folds=5, algorithm="ID3", dataset=dataset, k_neighbor=3)

    print("--------- first dataset 'Cancer' --------------")
    print("---------- Gaussian Naive Bayes ---------------")
    accuracy_naive = nb.evaluate_kfold()
    print("Naive Bayes Classification")
    print("Accuracy in each fold: %s" % accuracy_naive)
    print("Average Accuracy: %f" % (sum(accuracy_naive) / len(accuracy_naive)))
    1 / 0
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

    print("---------- Second dataset 'car' ---------------")
    print("---------- Gaussian Naive Bayes ---------------")
    accuracy_naive = evaluate_algorithm(dataset, naive_bayes, n_folds)
    print("Naive Bayes Classification")
    print("Accuracy in each fold: %s" % accuracy_naive)
    print("Average Accuracy: %f" % (sum(accuracy_naive) / len(accuracy_naive)))

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

    print("---------- third dataset 'ecol' ---------------")
    print("---------- Gaussian Naive Bayes ---------------")
    accuracy_naive = evaluate_algorithm(dataset, naive_bayes, n_folds)
    print("Naive Bayes Classification")
    print("Accuracy in each fold: %s" % accuracy_naive)
    print("Average Accuracy: %f" % (sum(accuracy_naive) / len(accuracy_naive)))

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

    print("---------- fourth dataset 'lett' --------------")
    print("---------- Gaussian Naive Bayes ---------------")
    # accuracy_naive = evaluate_algorithm(dataset, naive_bayes, n_folds)
    print("Naive Bayes Classification")
    print("Accuracy in each fold: %s" % accuracy_naive)
    print("Average Accuracy: %f" % (sum(accuracy_naive) / len(accuracy_naive)))

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

    print("---------- fifth dataset 'mush' --------------")
    print("---------- Gaussian Naive Bayes ---------------")
    accuracy_naive = evaluate_algorithm(dataset, naive_bayes, n_folds)
    print("Naive Bayes Classification")
    print("Accuracy in each fold: %s" % accuracy_naive)
    print("Average Accuracy: %f" % (sum(accuracy_naive) / len(accuracy_naive)))


if __name__ == "__main__":
    main()
