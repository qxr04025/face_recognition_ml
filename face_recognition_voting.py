#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===================================================================
Faces recognition example using eigenfaces and some emsemble methods
===================================================================

Author: qinxiaoran
"""

from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

###############################################################################
# Train a voting classification model
print("Fitting a voting classifier to the training set")
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
voting_clf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[1,1,3])
param_grid = {'dt__max_depth':range(3, 30, 5), 'knn__n_neighbors':range(3, 11, 2)}
voting_clf = GridSearchCV(estimator=voting_clf, param_grid=param_grid)
voting_clf = voting_clf.fit(X_train_pca, y_train)

###############################################################################
# Quantitative evaluation of the model quality on the test set
print("Predicting people's names on the test set")
origin = sys.stdout
fid = open('results_voting.txt', 'w')
sys.stdout = fid

print("***************Results of VotingClassifie******************")
print(classification_report(y_test, voting_clf.predict(X_test_pca), target_names=target_names))
print("accuracy:", voting_clf.score(X_test_pca, y_test))
print()

sys.stdout = origin
fid.close()