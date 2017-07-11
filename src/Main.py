from numpy import genfromtxt
import numpy as np
from Helper import list_difference, print_matrix, write_matrix
import random as rnd
import Helper
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import multioutput
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import os
import sys


class Main:

    def get_bool(self, prompt):
        while True:
            try:
                return {"true": True, "false": False}[input(prompt).lower()]
            except KeyError:
                print("Invalid input please enter True or False!")

    def select_two(self, m1, m2, bnd1, bnd2):
        """"" Select from two matricies and return the result, both boundaries exclusive

        Key Arguments:

        m1 -- first Matrix
        m2 -- second Matrix
        bnd1 -- boundary to select from within m1
        bnd2 -- boundary to select from within m2
        """
        bnd1=bnd1 -1
        bnd2=bnd2 -1

        slt1 = [rnd.randint(1, bnd1)]
        slt2 = [rnd.randint(1, bnd2)]

        X = [m1[slt1[0], :], m2[slt2[0], :]]
        # unmodified at i, modified at i+1
        sltd1 = 1
        sltd2 = 1

        for i in range(0, max(bnd1,bnd2)):
            next1 = rnd.randint(0, bnd1)
            next2 = rnd.randint(0, bnd2)

            if sltd1 > m1.shape[0]:
                slt1.clear()

            if sltd2 > m2.shape[0]:
                slt1.clear()

            while slt1.__contains__(next1) and slt2.__contains__(next2):
                next1 = rnd.randint(0, bnd1)
                next2 = rnd.randint(0, bnd2)

            if sltd1 < m1.shape[0]:
                X.extend(m1[next1])
                slt1.append(next1)
                sltd1=sltd1+1

            if sltd2 < m2.shape[0]:
                X.append(m2[next2])
                slt2.append(next2)
                sltd2=sltd2+1
        return X

    def make_binary(self, m):
        x = []
        for row in range(0, m.shape[0]):
            binary = 0
            for col in range(0, m.shape[1]):
                binary = binary + int(m[row, col]) * 10 ** (2-col)
            x.append(binary)
        return x

    def select_one(self, m, bnd):
        """""
            Selects from 0 - bnd and returns the result, exclusive
        '"""
        bnd = bnd - 1
        slt = []

        X = []

        for i in range(0, bnd):
            next_index = rnd.randint(0, bnd)
            while slt.__contains__(next_index):
                next_index = rnd.randint(0, bnd)
            slt.append(next_index)
            X.append(m[next_index])

        return X

    def execute(self):
        fileName = input("Enter file name: ")
        data = genfromtxt(fileName,dtype=str, delimiter=',')
        feature_data = genfromtxt('company_features.csv', dtype=str, delimiter=',')

        print("data: {0} feature_data: {1}".format(data.shape, feature_data.shape))
        concant = np.append(data, feature_data, axis=1)
        print(concant)
        #remember, [row, col]
        modified = data[0:910, :]
        unmodified = data[910:, :]

        rnd.seed(1234576)
        rnd.shuffle(modified)
        rnd.shuffle(unmodified)

        print("Unmodified: {0}".format(unmodified[-1, :]))
        print("Modified: {0}".format(modified[-1, :]))

        print("Unmodified Size: {0}".format(unmodified.shape[0]))
        print("Modified Size: {0}".format(modified.shape[0]))


        if self.get_bool("Create new data sets?"):
            bnd = int(input("Size of training set"))
            data_list = self.select_one(data, data.shape[0])
            rnd.shuffle(data_list)

            train_set_array = self.select_one(data_list, bnd)
            train_set = np.matrix(np.array(train_set_array))
            train_features = train_set[:, 0:]
            train_targets = train_set[:, 3:]

            test_set = np.matrix(np.array(Helper.list_difference(data_list, train_set_array)))
            test_features = test_set[:, 0:]
            test_targets = test_set[:, 3:6]

            write_matrix(train_targets, "train_targets.csv")
            write_matrix(train_features, "train_features.csv")

            write_matrix(test_targets, "test_targets.csv")
            write_matrix(test_features, "test_features.csv")
        else:
            train_features = genfromtxt('train_features.csv',dtype=float, delimiter=',')
            train_targets = genfromtxt('train_targets.csv',dtype=float, delimiter=',')

            test_targets = genfromtxt('test_targets.csv',dtype=float, delimiter=',')
            test_features = genfromtxt('test_features.csv',dtype=float, delimiter=',')

        print("Training Classifier")

        train_features = genfromtxt('test_targets.csv',dtype=float, delimiter=',')[1:, 1:]
        test_features = test_features[:, 1:]

        train_one_targets = np.array(self.make_binary(train_targets))
        test_one_targets = self.make_binary(test_targets)

        mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
        print(train_features.shape[0])
        print(len(train_one_targets))

        print(train_features)
        mlp.fit(train_features, train_one_targets)
        predictions = mlp.predict(test_features)

        print("Testing Classifier")
        print(classification_report(test_one_targets, predictions))



        # smplNum = int(input("Number to sample from both subsets for training?"))
        # X = self.select_two(unmodified, modified, smplNum, smplNum)
        # rnd.shuffle(X)

x = Main()
x.execute()