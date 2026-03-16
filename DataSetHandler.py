import numpy as np

class DataSetHandler:
    def __init__(self, sentence_data, train_percentage=0.8, seed=42):
        """
        DataSetHandler handles the splitting of the dataset into training and testing sets
        It also ensures that the training set is randomised each epoch
        :param sentence_data: numpy array of all sentences with words in index form
        :param train_percentage: percentage of dataset that is used for training
        :param seed: seed to ensure reproducibility of train and test set splitting
        """
        np.random.seed(seed)
        np.random.shuffle(sentence_data)
        self.training_set = sentence_data[:round(len(sentence_data) * train_percentage)]
        self.test_set = sentence_data[round(len(sentence_data) * train_percentage):]

    def get_training_set(self):
        """
        shuffles the training such that it can be used for training
        :return: shuffled numpy array of all sentences (in index form) in the training dataset
        """
        np.random.shuffle(self.training_set)
        return self.training_set

    def get_test_set(self):
        """
        getter for the test set
        :return: numpy array of all sentences (in index form) in the test dataset
        """
        return self.test_set