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
from keras.models import load_model
from sklearn.cross_validation import train_test_split

df = pd.read_csv('updated1l.csv')
cut_off = 80000
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
# integer encode the documents
# encoded_docs = t.texts_to_sequences(docs)

# embedding_matrix = zeros((vocab_size, 50))
# for word, i in t.word_index.items():
# 	# print(word, i)
# 	embedding_vector = embeddings_index.get(word)
# 	if embedding_vector is not None:
# 		embedding_matrix[i] = embedding_vector


data_split = 0.25

X_train, X_test, Y_train, Y_test = train_test_split(text, labels, test_size = data_split, random_state = 0)

print(len(X_train))
print(len(X_test))
print(len(Y_train))
print(len(Y_test))

lstm_out = 200

model = Sequential()
model.add(LSTM(lstm_out, input_shape = (None, 4000), dropout_U = 0.1, dropout_W = 0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
# compile the model
optimizer = keras.optimizers.Nadam()
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
input_train_size = int((1 - data_split) * cut_off)
input_test_size = int((data_split) * cut_off)

# print('input_size : ', input_size)
model.fit(X_train.reshape(input_train_size,1,4000), Y_train, epochs = 10, verbose = 1, batch_size = 4, validation_data = (X_test.reshape(input_test_size,1,4000), Y_test))


model.save('RNN_FINAL.model')