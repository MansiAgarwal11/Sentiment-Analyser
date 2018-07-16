from numpy import zeros
import keras
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
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import load_model
from sklearn.cross_validation import train_test_split

df = pd.read_csv('updated_30.csv', sep = '\t')
cut_off = 100
df = df[ : cut_off]

docs = list(map(str, df.text.values[ : cut_off]))
print(len(docs))
labels = list(df.label.values[ : cut_off])
negative = [1, 0, 0]
neutral = [0, 1, 0]
positive = [0, 0, 1]
# n = len(labels)
for i in range(cut_off):
	if(labels[i] == 0):
		labels[i] = copy.deepcopy(negative)
	elif(labels[i] == 1):
		labels[i] = copy.deepcopy(neutral)
	elif(labels[i] == 2):
		labels[i] = copy.deepcopy(positive)

labels = np.asarray(labels, dtype = 'uint64')
embeddings_index = dict()
f = open('./glove.6B.50d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

max_length = 400
padding = 4000
n = len(docs)
text = []
for i in range(n):
	# print(docs[i])
	l = docs[i].split(' ')
	m = len(l)
	res = []
	empty = [0 for i in range(50)]
	for j in range(m):
		if(l[j] in embeddings_index):
			res.extend(embeddings_index[l[j]])
		else:
			res.extend(empty)
	if(len(res) < 4000):
		extra = [0 for i in range(4000 - len(res))]
		res.extend(extra)
	text.append(res[ : 4000])

# print(text)
text = np.array(text)

print(text.shape)
print(len(labels))

t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1


data_split = 0.25

X_train, X_test, Y_train, Y_test = train_test_split(text, labels, test_size = data_split, random_state = 0)

print(len(X_train))
print(len(X_test))
print(len(Y_train))
print(len(Y_test))

model = Sequential()
# model.add(Flatten(input_shape = (1, 1, 4000)))
# model.add(Conv1D(4000, 3, activation='relu', input_shape = (1, 4000)))
model.add(Conv1D(filters = 4000, kernel_size = 4, strides = 1, padding = 'valid', input_shape = (4, 4000)))
model.add(MaxPooling1D(3))
# model.add(Conv1D(96, 3, activation='relu'))
# model.add(MaxPooling1D(3))
model.add(Conv1D(4000, 2, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(3, activation = 'softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.summary()

input_train_size = int((1 - data_split) * cut_off)
input_test_size = int((data_split) * cut_off)

print('input_train_size : ', input_train_size)
model.fit(X_train.reshape(input_train_size, 4, 1000), Y_train, epochs = 10, verbose = 1, batch_size = 1, validation_data = (X_test.reshape(input_test_size,1,4000), Y_test))



model.save('CNN_NEW.model')