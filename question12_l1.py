import bloscpack
X = bloscpack.unpack_ndarray_from_file("correct_context_que_X.blp")
y = bloscpack.unpack_ndarray_from_file("correct_context_que_Y.blp")

from sklearn.model_selection import train_test_split

X_train_original, X_test_original, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

breakpoint()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train_original)

X_train_original = scaler.transform(X_train_original)
X_test_original = scaler.transform(X_test_original)

from sklearn.linear_model import LogisticRegression
from halo import Halo
import pandas as pd
import matplotlib.pyplot as plt
import numpy


# question 12

c_list = [10 ** i for i in range(-5, 6)]

accuracies_by_c = list()
features_by_c = list()
zero_features_by_c = list()

for c in c_list:

    print("------------ C = " + str(c) + " ------------")

    accuracy_list= list()
    parameters_list = list()
    zero_features_list = list()
    nb_zero_parameters = 1  # juste pour l'initialiser à quelque chose
    X_train = X_train_original
    X_test = X_test_original

    while True:

        with Halo(text="Training", spinner='dots') as spinner:
            clf_l1 = LogisticRegression(penalty = 'l1', solver = 'saga', C=c)
            clf_l1.fit(X_train, y_train)
        spinner.succeed("Training")

        print("Number of features: " + str(X_train.shape[1]))
        parameters_list.append(X_train.shape[1])
        acc = clf_l1.score(X_test, y_test)
        accuracy_list.append(acc)

        mask = clf_l1.coef_[0] == 0

        if True not in mask:    # if there are no zero parameters
            print("Number of zero features: 0")
            print("Accuracy: " + str(acc))
            print("There are no zero parameters: break")
            break

        X_train = numpy.array(list(f[mask] for f in X_train))
        X_test = numpy.array(list(f[mask] for f in X_test))

        print("Number of zero features: " + str(X_train.shape[1]))
        print("Accuracy: " + str(acc))
        print()

        if nb_zero_parameters == X_train.shape[1]:  # si le nombre de parametres à zero n'a pas change
            print("The number of zero parameters has not changed: break")
            break
        nb_zero_parameters = X_train.shape[1]

    accuracies_by_c.append(accuracy_list)
    features_by_c.append(parameters_list)

print(accuracies_by_c)
print(features_by_c)
