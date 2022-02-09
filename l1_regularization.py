import bloscpack
X = bloscpack.unpack_ndarray_from_file("correct_context_que_X.blp")
y = bloscpack.unpack_ndarray_from_file("correct_context_que_Y.blp")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# scaling the features to converge faster (helps the solver)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from halo import Halo
import pandas as pd
import matplotlib.pyplot as plt

c_list = [10 ** i for i in range(-5, 6)]
#c_list = [10 ** i for i in range(-5, -1)]

accuracy_list_l1 = list()
#accuracy_list_l2 = list()
non_zero_parameters_list = list()
for c in c_list:

    print("------------ C = " + str(c) + " ------------")

    with Halo(text="Training l1", spinner='dots') as spinner:
        clf_l1 = LogisticRegression(penalty = 'l1', solver = 'saga', C=c)
        clf_l1.fit(X_train, y_train)
    spinner.succeed("Training l1")

    print(clf_l1.score(X_test, y_test))
    accuracy_list_l1.append(clf_l1.score(X_test, y_test))
    nb_non_zero_parameters = sum(x != 0 for x in clf_l1.coef_[0])
    non_zero_parameters_list.append(nb_non_zero_parameters)

    '''
    with Halo(text="Training l2", spinner='dots') as spinner:
        clf_l2 = LogisticRegression(penalty = 'l2', solver = 'saga', C=c)
        clf_l2.fit(X_train, y_train)
    spinner.succeed("Training l2")

    print(clf_l2.score(X_test, y_test))
    accuracy_list_l2.append(clf_l2.score(X_test, y_test))
    '''

'''
df1 = pd.DataFrame({"Value of C": c_list, "accuracy with l1": accuracy_list_l1, "accuracy with l2": accuracy_list_l2})
print(df1)

df1.plot(x = "Value of C", y = {"accuracy with l1", "accuracy with l2"}, ylabel = "Accuracy", style = '.-')
plt.xscale('log')
plt.show()
'''

df2 = pd.DataFrame({"Number of non-zero parameters": non_zero_parameters_list, "Accuracy": accuracy_list_l1})
print(df2)

df2.plot(x = "Number of non-zero parameters", ylabel = "Accuracy", style = '.-')
plt.show()

df3 = pd.DataFrame({"Value of C": c_list, "Number of non-zero parameters": non_zero_parameters_list})
print(df3)

df3.plot(x = "Value of C", ylabel = "Number of non-zero parameters", style = '.-')
plt.xscale('log')
plt.show()
