from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

import pandas as pd

from sklearn.externals import joblib

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

import random

classifier = XGBClassifier(eta = 0.005, max_depth = 10, min_child_weight = 1, gamma = 1.5, subsample = 1, colsample_bytree = 1, scale_pos_weight = 2.0, )
# classifier = XGBClassifier()



train_df = pd.read_csv('train.csv')    # Training data file here
text = list(train_df.text.values)
labels = list(train_df.label.values)
reviews = [labels, text]


vectorizer = CountVectorizer(encoding = 'utf-8')

train_features = vectorizer.fit_transform([str(reviews[1][i]) for i in range(len(reviews[1]))])
# test_features = vectorizer.transform([str(test[1][i]) for i in range(len(test[1]))])

X_train, X_test, Y_train, Y_test = train_test_split(train_features, [reviews[0][i] for i in range(len(reviews[0]))], test_size = 0.3, random_state = 1)
# X_train, Y_train = (train_features, [reviews[0][i] for i in range(len(reviews[0]))])




# classifier = AdaBoostClassifier()
# classifier = GradientBoostingClassifier(n_estimators = 200, max_depth = 2, max_leaf_nodes = 4)

classifier.fit(X_train, Y_train)
x = classifier.score(X_test, Y_test)
print(x)
joblib.dump(classifier, 'XG_test.pkl')

# X_test, Y_test = (X_train, Y_train)


