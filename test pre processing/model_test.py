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



def text_process(docs, max_length, start, end):
	dataset = docs
	count = 0
	corpus = []
	for i in range(start, end):
		review = re.sub('[^a-zA-Z]', ' ', dataset[i]) #" " replaces the chars other than alphabets
		gs = goslate.Goslate()
		print(count)
		try:
		    review = gs.translate(review,'en')
		except:
		    pass
		review = review.lower()
		#review is a string, split converts the string into words by using space as seperator
		ps = PorterStemmer()
		Lemma = WordNetLemmatizer()
		#set used for faster execution than list
		# review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
		#to convert back review from list to string use join
		# review = review.split()
		review = [Lemma.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
		review = ' '.join(review)
		corpus.append(review)
		count += 1
		l = range(start, i + 1)
		df = pd.DataFrame({'ID' : l, 'text' : corpus})
		df.to_csv('updated_2.csv', sep = '\t', encoding = 'utf-8', index = False)

	docs = corpus
	t = Tokenizer()
	t.fit_on_texts(docs)
	vocab_size = len(t.word_index) + 1
	encoded_docs = t.texts_to_sequences(docs)
	# max_length = 400
	padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	padded_docs = np.asarray(padded_docs)
	return padded_docs

def csv_process(csv_filename):
	df = pd.read_csv(csv_filename)
	n = 100
	# ID = df.ID.values[ : n]
	# text = df.text.values[ : n]
	ID = df.ID.values
	docs = list(map(str, df.text.values))
	return (ID, docs)

start = 0
end = 25000


csv_filename = 'test_data.csv'
max_length = 500
ID, text = csv_process(csv_filename)
text = text_process(text, max_length, start, end)


model_filename = 'CNN_30k.model'
model = load_model(model_filename)
model.summary()
result = [[], []]
index = 0

for i in text:
	probability = model.predict(i.reshape(1, max_length))
	# probability = list(probability)
	print(probability)
	m = probability.max()
	print('max : ', m)
	l = probability.tolist()
	if(len(l) == 1):
		l = l[0]
	label = l.index(m)
	result[1].append(label)
	result[0].append(ID[index])
	index += 1

df = pd.DataFrame({'ID' : result[0], 'label' : result[1]})
df.to_csv('result.csv', sep = ',', encoding = 'utf8', index = False)
