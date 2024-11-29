from sklearn.model_selection import train_test_split
import utils
import config
from diffprivlib.models import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

epsilons = np.logspace(-2, 2, 50)
bounds = ([4, 0], [1234, 3456])
accuracy = list()
_,_,_,_,egs=utils.read_graph(config.train_file)
print(len(egs))
X_train, X_test = train_test_split(egs, test_size=0.3)
y=[0 for x in range(len(X_train))]
y1=[0 for x in range(len(X_test))]
print("done")
print(len(X_train))
print(len(X_test))
print(len(X_test)+len(X_train))
clf = GaussianNB()
clf.fit(X_train,y)
clf.predict(X_test)
print("Test accuracy: %f" % clf.score(X_test,y1))
for epsilon in epsilons:
    clf = GaussianNB(bounds=bounds, epsilon=epsilon)
    clf.fit(X_train,y)

    accuracy.append(clf.score(X_test,y1))
print(accuracy)