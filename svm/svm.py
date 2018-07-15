from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from numpy import asarray 
import copy
import pandas as pd
import numpy as np
import random
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from gooimport re
import nltk
nltk.download('stopwords') #stopwords contain all the irrelevant words
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import goslate
from googletrans import Translator
import re
import nltk
nltk.download('stopwords') #stopwords contain all the irrelevant words
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import goslate
from sklearn import svm
from sklearn.externals import joblib

clf = svm.SVC(gamma=0.001, C=100.0)
df = pd.read_csv('train.csv')
# df = df[ : 1000]
# docs = df.text.values
train_len = 80000

docs = list(map(str, df.text.values))

labels = df.label.values

vectorizer = CountVectorizer(encoding = 'utf-8')

train_features = vectorizer.fit_transform([str(reviews[1][i]) for i in range(len(reviews[1]))])
# test_features = vectorizer.transform([str(test[1][i]) for i in range(len(test[1]))])

X_train, X_test, Y_train, Y_test = train_test_split(train_features, [reviews[0][i] for i in range(len(reviews[0]))], test_size = 0.3, random_state = 5)

clf.fit(X_train, Y_train)
n = 10000

score = clf.score(X_test, Y_test)

print(score)

joblib.dump(clf, 'svm_BIG.pkl')