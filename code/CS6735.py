import os
import numpy as np
import pandas as pd
from random import randrange
import csv
import math, copy
import operator
from collections import Counter
import pprint as pp


class classifiers:
    def __init__(
        self,
        n_folds,
        dataset,
        algorithm,
    ):
        self.n_folds = n_folds
        self.dataset = dataset
        self.algorithm = algorithm

    def kfold_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_duplicated = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
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

    def fit(self):
        folds = self.kfold_split(self.dataset, self.n_folds)
        scores = list()
        for fold in folds:
            trainset = list(folds)
            trainset.remove(fold)
            trainset = sum(trainset, [])
            testset = list()
            for row in fold:
                row_copy = list(row)
                testset.append(row_copy)
            # print(testset)
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


class NB(classifiers):
    def __init__(
        self,
        n_folds,
        dataset,
        verbose=True,
    ):
        self.n_folds = n_folds
        self.dataset = dataset
        self.verbose = verbose
        self.algorithm = self.naivebayes
        classifiers(n_folds, dataset, self.algorithm)

    def naivebayes(self, trainset, testset):
        model = self.model_classes(trainset)
        predictions = []
        for i in range(len(testset)):
            result = self.predict(model, testset[i])
            predictions.append(result)
        return predictions

    def predict(self, models, test_row):
        probabilities = {}
        for (classValue, classModels) in models.items():
            probabilities[classValue] = 1
            for i in range(len(classModels)):
                (mean, stdev) = classModels[i]
                x = test_row[i]
                probabilities[classValue] *= self.find_pdf(x, mean, stdev)

        (bestLabel, bestProb) = (None, -1)
        for (classValue, probability) in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def model_classes(self, dataset):
        separatedbyclasses = {}
        for i in range(len(dataset)):
            row = dataset[i]
            if row[-1] not in separatedbyclasses:
                separatedbyclasses[row[-1]] = []
            separatedbyclasses[row[-1]].append(row)

        classmodels = {}

        for (classValue, instances) in separatedbyclasses.items():
            models = list()
            for feature in zip(*instances):
                models.append((np.mean(feature), np.std(feature)))

            classmodels[classValue] = models[:-1]

        return classmodels

    def find_pdf(self, x, mean, stdev):
        if stdev == 0.0:
            if x == mean:
                return 1.0
            else:
                return 0.0
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return 1 / (math.sqrt(2 * math.pi) * stdev) * exponent


class KNN(classifiers):
    def __init__(
        self,
        n_folds,
        dataset,
        k_neighbor=3,
        verbose=False,
    ):
        self.n_folds = n_folds
        self.k_neighbor = k_neighbor
        self.dataset = dataset
        self.verbose = verbose
        self.algorithm = self.knn
        classifiers(n_folds, dataset, self.algorithm)

    def find_response(self, neighbors, classes):
        votes = [0] * len(classes)

        for instance in neighbors:

            for ctr, c in enumerate(classes):
                if instance[-2] == c:
                    votes[ctr] += 1

        return max(enumerate(votes), key=operator.itemgetter(1))

    def knn(self, training_set, test_set):
        distances = []
        features_number = len(training_set[0]) - 1

        classes = list(set([row[-1] for row in training_set]))
        prediction = list()
        for _, test_instance in enumerate(test_set):

            for row in training_set:
                temp = math.dist(row[:features_number], test_instance[:features_number])

                distances.append(row + [temp])

            distances.sort(key=operator.itemgetter(len(distances[0]) - 1))

            neighbors = distances[0 : self.k_neighbor]

            index, value = self.find_response(neighbors, classes)

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


class ID3(classifiers):
    def __init__(
        self,
        n_folds,
        dataset,
        names,
        verbose=False,
    ):
        self.n_folds = n_folds
        self.dataset = dataset
        self.verbose = verbose
        self.names = names
        self.algorithm = self.id3
        classifiers(n_folds, dataset, self.algorithm)

    def predict(self, query, tree, default=1):
        for key in list(query.keys()):
            if key in list(tree.keys()):
                try:
                    result = tree[key][query[key]]
                except:
                    return default
                result = tree[key][query[key]]
                if isinstance(result, dict):
                    return self.predict(query, result)
                else:
                    return result

    def test(self, data, tree, features_name):
        Df = pd.DataFrame(data, columns=features_name)
        queries = Df.iloc[:, :-1].to_dict(orient="records")
        predicted = pd.DataFrame(columns=["predicted"])
        for i in range(len(data)):
            predicted.loc[i, "predicted"] = self.predict(queries[i], tree, 4.0)
        return predicted

    def entropy(self, data):
        entries = len(data)
        labels = {}
        for row in data:
            label = row[-1]
            if label not in labels.keys():
                labels[label] = 0
            labels[label] += 1

        entropy = 0.0
        for key in labels:
            prob = float(labels[key]) / entries
            entropy -= prob * math.log(prob, 2)
        return entropy

    def attr_selection(self, data):
        features = len(data[0]) - 1
        baseEntropy = self.entropy(data)
        max_InfoGain = 0.0
        bestAttr = -1
        for i in range(features):
            AttrList = [row[i] for row in data]
            uniqueVals = set(AttrList)

            newEntropy = 0.0
            attrEntropy = 0.0
            for value in uniqueVals:
                newData = self.split(data, i, value)
                prob = len(newData) / float(len(data))
                newEntropy = prob * self.entropy(newData)
                attrEntropy += newEntropy
            infoGain = baseEntropy - attrEntropy
            if infoGain > max_InfoGain:
                max_InfoGain = infoGain
                bestAttr = i
        return bestAttr

    def split(self, data, arc, val):
        newData = []
        for row in data:
            if row[arc] == val:
                reducedSet = list(row[:arc])
                reducedSet.extend(row[arc + 1 :])
                newData.append(reducedSet)
        return newData

    def decision_tree(self, data, label):  # ,flag):
        labels = copy.deepcopy(label)
        classList = [row[-1] for row in data]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        maxGainNode = self.attr_selection(data)
        treeLabel = labels[maxGainNode]
        theTree = {treeLabel: {}}
        del labels[maxGainNode]
        nodeValues = [row[maxGainNode] for row in data]
        uniqueVals = set(nodeValues)
        for value in uniqueVals:
            subLabels = labels[:]
            theTree[treeLabel][value] = self.decision_tree(
                self.split(data, maxGainNode, value), subLabels
            )
        return theTree

    def id3(self, train_set, test_set):
        tree = self.decision_tree(train_set, self.names)
        # pp.pprint(tree)
        predicted = self.test(test_set, tree, self.names)
        return predicted.values.T.squeeze()


class RF(ID3):
    num_trees = 0
    decision_trees = []
    bootstraps_datasets = []
    bootstraps_labels = []

    def __init__(
        self,
        n_folds,
        dataset,
        names,
        num_trees=5,
        verbose=True,
    ):
        self.n_folds = n_folds
        self.dataset = dataset
        self.num_trees = num_trees
        self.verbose = verbose
        self.names = names
        self.algorithm = self.RF
        classifiers(n_folds, dataset, self.algorithm)

    def bootstrapping(self, training_set):
        samples = []
        for _ in range(len(training_set)):
            randomindex = randrange(0, len(training_set))
            samples.append(training_set[randomindex][:])
        return samples

    def RF(self, training_set, test_set):
        predictions = list()
        for _ in range(self.num_trees):
            data_sample = self.bootstrapping(training_set)
            predictions.append(self.id3(data_sample, test_set))
        y = self.voting(predictions)
        return y

    def voting(self, predictions):
        y = list()
        for sample in zip(*predictions):
            q = Counter(sample)
            y.append(q.most_common(1)[0][0])
        return y


class stump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class AB(classifiers):
    def __init__(self, n_folds, dataset, n_clf=5):
        self.n_clf = n_clf
        self.n_folds = n_folds
        self.dataset = dataset
        self.algorithm = self.AB
        classifiers(n_folds, dataset, self.algorithm)

    def AB(self, trainset, testset):
        X = list()
        y = list()
        for row in trainset:
            X.append(row[:-1])
            y.append(row[-1])
        self.ABoost(np.array(X), np.array(y))
        y_pred = self.predict(np.array(testset))
        return y_pred

    def ABoost(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            clf = stump()

            min_error = float("inf")
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            predictions = clf.predict(X)

            w *= np.exp(-1 * clf.alpha * y * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred


class files:
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


def discr(x, k):
    w = (np.max(x) - np.min(x)) / k
    bins = [np.min(x) + (i) * w for i in range(k)]
    return np.digitize(x, bins=bins, right=False)