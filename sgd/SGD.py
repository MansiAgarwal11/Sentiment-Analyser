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

from sklearn.linear_model import SGDClassifier

import random

train_df = pd.read_csv('train_processed.csv', sep = ',')    # Training data file here
text = list(train_df.text.values)
labels = list(train_df.label.values)
reviews = [labels, text]


clf = SGDClassifier(warm_start = True, learning_rate = 'invscaling', eta0 = 0.06, power_t = 0.3)


vectorizer = CountVectorizer(encoding = 'utf-8', max_features = 75000)

train_features = vectorizer.fit_transform([str(reviews[1][i]) for i in range(len(reviews[1]))])
# test_features = vectorizer.transform([str(test[1][i]) for i in range(len(test[1]))])

X_train, X_test, Y_train, Y_test = train_test_split(train_features, [reviews[0][i] for i in range(len(reviews[0]))], test_size = 0.3, random_state = random.randint(0, 1000))

# clf = SGDClassifier(warm_start = True, learning_rate = 'invscaling', eta0 = 0.5)

clf.fit(X_train, Y_train)

joblib.dump(clf, 'SGD_big_newest.pkl')
score = clf.score(X_test, Y_test)
print(score)
exit()
# print('test data')
# train_df = pd.read_csv('test_data.csv', sep = ',')    # Training data file here
# text = list(train_df.text.values)
# ID = list(train_df.ID.values)
# reviews = [ID, text]
# print('transform')
# test_features = vectorizer.fit_transform([str(reviews[1][i]) for i in range(len(reviews[1]))])

# score = clf.predict(test_features)
# label = []
# ID = list(range(1, len(ID) + 1))
# label.extend(score)
# print('saving')
# print('ID : ', len(ID))
# print('label : ', len(label))
# df = pd.DataFrame({'ID' : ID, 'label' : label})
# df.to_csv('./result_SGD.csv', sep = ',', encoding = 'utf8', index = False)

# print('updated_final')

# train_df = pd.read_csv('updated_final.csv', sep = '\t')    # Training data file here
# text = list(train_df.text.values)
# ID = list(train_df.ID.values)
# reviews = [ID, text]
# print('transform')
# test_features = vectorizer.fit_transform([str(reviews[1][i]) for i in range(len(reviews[1]))])

# score = clf.predict(test_features)

# label = []
# ID = list(range(1, len(ID) + 1))
# label.extend(score)
# print('saving')
# print('ID : ', len(ID))
# print('label : ', len(label))

# df = pd.DataFrame({'ID' : ID, 'label' : label})
# df.to_csv('./result_SGD_2.csv', sep = ',', encoding = 'utf8', index = False)


