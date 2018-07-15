from  collections import Counter
import csv
import re
import pandas as pf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

import time

# count = 0

# counter = 0

# with open('./Datasets/updated(0-1L).csv', 'r') as file:
# 	reviews = list(csv.reader(file))

# # with open('train_subset.csv', 'w') as train_subset_file:
# # 	writer = csv.writer(train_subset_file)
# # 	for review in reviews:
# # 		if review[0] == 'label':
# # 			continue
# # 		if count>=33000:
# # 			break
# # 		writer.writerow(review)
# # 		count+=1

# # count = 0

# with open('test_subset.csv', 'w') as test_dataset_file:
# 	writer = csv.writer(test_dataset_file)
# 	for review in reviews:
# 		if review[0] == 'label':
# 			continue
# 		if count<88600:
# 			count+=1
# 			continue
# 		if count>=89600:
# 			break
# 		writer.writerow(review)
# 		count+=1 


# with open('./updated_training_subset.csv', 'r') as file:
# 	reviews = list(csv.reader(file))

train_df = pf.read_csv('./Datasets/train.csv', low_memory = False)    # Training data file here
train_df = train_df.head(390000)
text = list(train_df.text.values)
labels = list(train_df.label.values)
reviews = [labels, text]

print(len(reviews[0]))

# ''' for x in reviews:
# 	print(x)
# 	print() '''

# # print(reviews[1][5], "\t", reviews[0][5])

# # for i in range(len(reviews[0])):
# # 	print(reviews[1][i]reviews[0][i])
# # 	print('\n\n')

# def get_text(reviews, score):
# 	# for i in range(len(reviews[0])):
# 	# 	print(reviews[0][i] == score)
# 	# 	print('\n\n')
# 	return " ".join(str(reviews[1][i]) for i in range(len(reviews[0])) if reviews[0][i] == score)

# def count_text(text):
# 	words = re.split("\s+", text)
# 	return Counter(words)

# negative_text = get_text(reviews, 0)
# neutral_text = get_text(reviews, 1)
# positive_text = get_text(reviews, 2)

# # print("Negative text: ", negative_text)

# negative_counts = count_text(negative_text)
# neutral_counts = count_text(neutral_text)
# positive_counts = count_text(positive_text)

# print("Counts: ", negative_counts, "\n\n", neutral_counts, "\n\n", positive_counts)

# def get_y_count(score):
# 	return len([reviews[1][i] for i in range(len(reviews[0])) if reviews[0][i] == score])

# negative_review_count = get_y_count(0)
# neutral_review_count = get_y_count(1)
# positive_review_count = get_y_count(2)

# print(negative_review_count, "\t", neutral_review_count, "\t", positive_review_count)

# prob_positive = positive_review_count/len(reviews[0])
# prob_neutral = neutral_review_count/len(reviews[0])
# prob_negative = negative_review_count/len(reviews[0])

# print("prob_negative: ", prob_negative, "\nprob_neutral: ", prob_neutral, "\nprob_positive: ", prob_positive)

# def make_class_prediction(text, counts, class_prob, class_count):
# 	prediction = 1
# 	text_counts = Counter(re.split("\s+", text))
# 	for word in text_counts:
# 		# print("Word: ", word)
# 		prediction *= text_counts.get(word) * ((counts.get(word, 0) + 1)/(sum(counts.values()) + class_count))

# 	return prediction * class_prob


# print("Review: ", reviews[1][3])
# print("Negative prediction: ", make_class_prediction(reviews[1][3], negative_counts, prob_negative, negative_review_count))
# print("Neutral prediction: ", make_class_prediction(reviews[1][3], neutral_counts, prob_neutral, neutral_review_count))
# print("Positive prediction: ", make_class_prediction(reviews[1][3], positive_counts, prob_positive, positive_review_count))


# def make_decision(text, make_class_prediction):
# 	negative_prediction = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
# 	neutral_prediction = make_class_prediction(text, neutral_counts, prob_neutral, neutral_review_count)
# 	positive_prediction = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)

# 	global counter
# 	counter+=1
# 	print(counter)

# 	if max([negative_prediction, neutral_prediction, positive_prediction]) == negative_prediction:
# 		return 0
# 	elif max([negative_prediction, neutral_prediction, positive_prediction]) == neutral_prediction:
# 		return 1
# 	return 2

# # print()

# # with open('./updated_test_dataset.csv', 'r') as file:
# # 	test = list(csv.reader(file))

test_df = pf.read_csv('./Datasets/train.csv', low_memory = False)  # Test data file here
test_df = test_df.tail(10000)
text = list(test_df.text.values)
labels = list(test_df.label.values)
test = [labels, text]

# print(len(test[0]))

# # print(len(test[0]))

# predictions = [make_decision(str(test[1][i]), make_class_prediction) for i in range(len(test[0])) if test[0][i] != 'label']

# print("Number of predictions: ", len(predictions))
# for x in predictions:
# 	print(x)


# Calculating error

actual = [test[0][i] for i in range(len(test[0]))]

# no_correct = len([ actual[i] for i in range(len(predictions)) if actual[i] == predictions[i] ])
# print("Percentage correct: ", no_correct/len(actual))



#########################################################################################################

# using sklearn NB

#########################################################################################################

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

# vectorizer = CountVectorizer(stop_words = 'english', encoding = 'utf-8')

vectorizer = CountVectorizer(ngram_range = (2, 2))															

train_features = vectorizer.fit_transform([str(reviews[1][i]) for i in range(len(reviews[1]))])
test_features = vectorizer.transform([str(test[1][i]) for i in range(len(test[1]))])

nb = MultinomialNB()
nb.fit(train_features, [reviews[0][i] for i in range(len(reviews[0]))])

sklearn_predict = nb.predict(test_features)

# output_df = pf.DataFrame({'ID': np.arange(1, len(test[0])+1), 'label': sklearn_predict})

# print(output_df)

# output_df.to_csv('./output_file_4.csv', index = False)

print("Finished")

print(len(sklearn_predict))

no_correct = len([ actual[i] for i in range(len(sklearn_predict)) if actual[i] == sklearn_predict[i] ])
print("Percentage correct: ", no_correct/len(actual))






