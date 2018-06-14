# -*- coding: utf-8 -*-
# Zadanie 1 (7 pkt.)
"""
Kod muszą państwo zaimplementować w pliku `assignment_L6_1.py`, a gotowe zadanie oddajemy wypychając zmiany na repozytorium.

+ Załaduj zbiór danych __iris__ korzystając z funkcji [load_iris](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
+ Korzystając z funkcji [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) oraz 
kryterium [mutual_info_classif](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif) 
wybierz najlepsze __dwa__ atrybuty 
+ Korzystając z [tego](http://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html) przykładu wyświetl na 
jednym wykresie granice decyzyjne dla następujących klasyfikatorów:
 + KNN z liczbą najbliższych sąsiadów 1;
 + Liniowy SVM;
 + SVM z jądrem RBF;
 + Naive Bayes;
 + Drzewa dacyzyjnego o maksymalnej głębokosci 10.
 
"""
from itertools import product
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB


iris = datasets.load_iris()

X = iris.data[:, [0, 2]]
y = iris.target

selector = SelectKBest(mutual_info_classif, k=2)
selector.fit(X, y)

X_new = selector.transform(X)

clf1 = DecisionTreeClassifier(max_depth=10)
clf2 = KNeighborsClassifier(n_neighbors=1)
clf3 = SVC(kernel='rbf', probability=True)
clf4 = SVC(kernel='linear', probability=True)
clf5 = GaussianNB()
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc1', clf3), ('svc2', clf4), ('gnb', clf5)],
                        voting='soft', weights=[2, 1, 2, 2, 1])

clf1.fit(X_new, y)
clf2.fit(X_new, y)
clf3.fit(X_new, y)
clf4.fit(X_new, y)
clf5.fit(X_new, y)
eclf.fit(X_new, y)

x_min, x_max = X_new[:, 0].min() - 1, X_new[:, 0].max() + 1
y_min, y_max = X_new[:, 1].min() - 1, X_new[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1], [0, 1]),
                        [clf1, clf2, clf3, clf4, clf5, eclf],
                        ['Decision Tree (depth=10)', 'KNN (k=1)',
                         'Kernel SVM(rbf)', 'Kernel SVM(linear)', 'GaussianNB()' 'Soft Voting']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X_new[:, 0], X_new[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()

