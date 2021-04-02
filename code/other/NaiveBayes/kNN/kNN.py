import csv
import sys
import math
import operator


def load_data_set(filename):
    print(filename)
    with open(filename, "r") as handle:
        return list(csv.reader(handle, delimiter=","))


def convert_to_float(data_set, mode):
    new_set = []
    if mode == "training":
        for data in data_set:
            new_set.append(
                [float(x) for x in data[: len(data) - 1]] + [data[len(data) - 1]]
            )

    elif mode == "test":
        for data in data_set:
            new_set.append([float(x) for x in data])

    else:
        print("Invalid mode, program will exit.")
        sys.exit()

    return new_set


def get_classes(training_set):
    return list(set([c[-1] for c in training_set]))


def find_neighbors(distances, k):
    return distances[0:k]


def find_response(neighbors, classes):
    votes = [0] * len(classes)

    for instance in neighbors:
        for ctr, c in enumerate(classes):
            if instance[-2] == c:
                votes[ctr] += 1

    return max(enumerate(votes), key=operator.itemgetter(1))


def knn(training_set, test_set, k):
    distances = []
    dist = 0

    limit = len(training_set[0]) - 1

    # generate response classes from training data
    classes = get_classes(training_set)

    for test_instance in test_set:
        for row in training_set:
            for x, y in zip(row[:limit], test_instance):
                dist += (x - y) * (x - y)

            distances.append(row + [math.sqrt(dist)])
            dist = 0

        distances.sort(key=operator.itemgetter(len(distances[0]) - 1))

        # find k nearest neighbors
        neighbors = find_neighbors(distances, k)

        # get the class with maximum votes
        index, value = find_response(neighbors, classes)

        # Display prediction
        print(
            "The predicted class for sample "
            + str(test_instance)
            + " is : "
            + classes[index]
        )
        print("Number of votes : " + str(value) + " out of " + str(k))

        # empty the distance list
        distances.clear()


def main():
    # get value of k
    k = 2  # int(input('Enter the value of k : '))

    # load the training and test data set
    training_file = "/Users/saeedkazemi/Documents/Python/CS6735/code/kNN/iris-dataset.csv"  # code/kNN/iris-dataset.csv"  # input('Enter name of training data file : ')
    test_file = "/Users/saeedkazemi/Documents/Python/CS6735/code/kNN/iris-test.csv"  # "code/kNN/iris-teat.csv"  # input('Enter name of test data file : ')

    training_set = convert_to_float(load_data_set(training_file), "training")
    test_set = convert_to_float(load_data_set(test_file), "test")

    if not training_set:
        print("Empty training set")

    elif not test_set:
        print("Empty test set")

    elif k > len(training_set):
        print(
            "Expected number of neighbors is higher than number of training data instances"
        )

    else:
        knn(training_set, test_set, k)


if __name__ == "__main__":
    main()