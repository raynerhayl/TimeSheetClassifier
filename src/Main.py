from numpy import genfromtxt
import numpy as np
from Helper import list_difference, print_matrix
import random as rnd
import Helper
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix



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


        if self.get_bool("Use entire data set?"):
            bnd = int(input("Size of training set"))
            data_list = self.select_one(data, data.shape[0])
            rnd.shuffle(data_list)

            print("1")
            train_set_array = self.select_one(data_list, bnd)
            train_set = np.matrix(np.array(train_set_array))
            train_features = train_set[:, 0:3]
            train_targets = train_set[:, 3:]

            print("2")

            test_set = np.matrix(np.array(Helper.list_difference(data_list, train_set_array)))

            print("3")

            test_features = test_set[:, 0:3]
            test_targets = test_set[:, 3:6]

            print("Train Features")
            print(train_features)

            print("Train Targets")
            print(train_targets)

            print("Test Features")
            print(test_features)

            print("Test Targets")
            print(test_targets)

            print("Training Classifier")
            mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
            mlp.fit(train_features, train_targets)

            print("Testing Classifier")
            predictions = mlp.predict(test_features)
            print(classification_report(test_targets, predictions))


        else:
            smplNum = int(input("Number to sample from both subsets for training?"))
            X = self.select_two(unmodified, modified, smplNum, smplNum)
            rnd.shuffle(X)

        print_matrix(X)


x = Main()
x.execute()