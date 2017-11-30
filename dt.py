"""
dt contains an ID3 decision-tree implementation, a helper function to evaluate
the tree, a function that splits data into test and training data, and a sample
of a decision tree being built on data and evaluated.
"""

import collections
import math
import random
import time


def evaluate(dt, test_name):
    """
    evaluate the DecisionTree's accuracy on test-data in test_name.

    Accuracy is reported as 100*(# classified correctly)/(total).
    """
    correct = 0
    total = 0
    with open(test_name) as f:
        for row in f:
            row = [col.strip() for col in row.split(',')]
            point, actual = row[:-1], row[-1]
            total += 1
            try:
                if dt.classify(point) == actual:
                    correct += 1
            except ValueError:
                # Didn't know how to classify the instance. Means the training
                # data didn't fully cover the attribtue space.
                pass
    return 100*correct/total


def info(data):
    """info is information inherent in data."""
    labels = data[-1]
    total = 0
    for frequency in collections.Counter(labels).values():
        ratio = frequency / len(labels)
        total -= ratio*math.log2(ratio)
    return total


def info_attribute(data, attribute):
    """info_attribute is the information of the data given the attribute."""
    values = data[attribute]
    total = 0
    for value, frequency in collections.Counter(values).items():
        ratio = frequency / len(values)
        split = [[] for _ in data]
        for row in zip(*data):
            if row[attribute] == value:
                for (i, col) in enumerate(row):
                    split[i].append(col)
        total += ratio*info(split)
    return total


def gain(data, attribute):
    """gain of information by splitting data on attribute."""
    return info(data) - info_attribute(data, attribute)


def highest_information_gain(data, used):
    """
    _highest_information_gain returns the index of the highest information gain
    attribute that isn't already in used in data.
    """
    remaining = set(range(len(data[:-1]))) - used
    highest = None
    index = None
    for attribute in remaining:
        gained = gain(data, attribute)
        if highest is None or gained > highest:
            highest = gained
            index = attribute
    return attribute


class Node(object):
    """
    Node is a DecisionTree Node that contains either the classification
    decision or information on how to traverse the tree.
    """

    def __init__(self, data, used):
        """
        __init__ recursively constructs a Node out of data by splitting on
        attributes not in used.
        """
        # Initialize the node.
        self.__index = 0 # Index the Node splits on if not a leaf.
        self.__branches = {} # Children Nodes.
        self.__value = '' # Value of the Node if it is a leaf.

        labels = data[-1]

        # Base cases.
        if len(set(labels)) == 1:
            # Every element has the same label so assign it.
            self.__value = labels[0]
            return
        unsplit_attributes = set(range(len(data)-1)) - used
        if len(unsplit_attributes) == 1:
            # No more attributes left to select from. Assign this leaf the label
            # that occurs the most frequently.
            self.__value = collections.Counter(labels).most_common()[0][0]
            return

        # In the recursive case, split on the attribute with highest information
        # gain and add children Nodes which are the result of splitting the
        # data on the attribute.
        highest_gain = highest_information_gain(data, used)
        # Index of the split attribute.
        self.__index = highest_gain
        for value in set(data[highest_gain]):
            # Partition a new child_data that is the result of splitting on the
            # attribute with the highest gain.
            child_data = [[] for _ in data]
            for row in zip(*data):
                if row[highest_gain] == value:
                    for (i, col) in enumerate(row):
                        child_data[i].append(col)
            self.__branches[value] = Node(child_data, used|{highest_gain})

    def classify(self, point):
        """
        classify returns the Node's value if it is a leaf and recurses down
        otherwise.
        """
        if len(self.__branches) == 0:
            # Return the value if at a leaf.
            return self.__value
        if point[self.__index] not in self.__branches:
            raise ValueError("don't know how to classify Point")
        # Otherwise, follow the decision tree until a leaf.
        return self.__branches[point[self.__index]].classify(point)


class DecisionTree(object):
    """
    DecisionTree is a classifier which uses decision-tree learning.

    data_name is the training-data to use. Is expected to be in a CSV format
    with the attribute to be classified as the last column. All attributes are
    expected to be discretized.
    """

    def __init__(self, data_name):
        """
        __init__ constructs the DecisionTree by parsing data in data_name and
        building a root DecisionTree Node.
        """
        # Parse the data.
        data = None
        with open(data_name) as f:
            for row in f:
                row = [col.strip() for col in row.split(',')]
                if data is None: data = [[] for _ in row]
                for i, col in enumerate(row): data[i].append(col)
        # Build the tree with no attributes used initially.
        self.__root = Node(data, set())

    def classify(self, point):
        """classify the point into a class."""
        return self.__root.classify(point)


def split(data_name, train_ratio):
    """
    split the data in file at data_name into data_name.train with
    train_ratio*number_of_rows and data_name.test with
    (1-train_ratio)*number_of_rows.

    Returns the names of the created training and test files.
    """
    rows = []
    with open(data_name) as f:
        for row in f: rows.append(row)
    random.shuffle(rows)
    train = rows[:math.ceil(len(rows)*train_ratio)]
    test = rows[math.ceil(len(rows)*train_ratio):]
    with open(data_name + '.train', 'w') as f:
        for row in train: f.write(row)
    with open(data_name + '.test', 'w') as f:
        for row in test: f.write(row)
    return data_name + '.train', data_name + '.test'


def run(data, training_ratio):
    """
    run the sample on the provided data file with the provided training ratio.

    Prints run-times and phases and returns the evaluation score.
    """
    train, test = split(data, training_ratio)
    print('Starting Training')
    now = time.time()
    dt = DecisionTree(train)
    print('Finished Training in {} seconds'.format(time.time()-now))
    print('Starting Evaluation')
    now = time.time()
    score = evaluate(dt, test)
    print('Finished Evaluating in {} seconds'.format(time.time()-now))
    return score


if __name__ == '__main__':
    DATA = 'car.data'
    TRAINING_RATIO = 0.75
    random.seed(time.time())

    for i in range(2):
        score = run(DATA, TRAINING_RATIO)
        print('Evaluation {} (100*correct/total): {}'.format(i+1, score))
        print()
