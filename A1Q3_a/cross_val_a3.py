# Question 3 (a)(iii)
# http://axon.cs.byu.edu/Dan/478/assignments/permutation_test.php

import numpy as np
from os import listdir
import tensorflow as tf
import tensorflow.keras
from nltk.tokenize import word_tokenize
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import h5py

def file_read_lexicon(file_path):
	#input type: str
	#output type: list
	f = open(file_path, 'r', encoding='ISO-8859-1')
	file_content = f.read()
	lexicon_list = file_content.split('\n')
	lexicon_list.pop()
	return lexicon_list

def unison_shuffle(a,b,c):
	assert(len(a)==len(b) and len(a) == len(c))
	p = np.random.permutation(len(a))
	return a[p], b[p], c[p]

def model_architecture(input_shape):
	model = Sequential()
	model.add(Dense(300, activation='relu', input_shape=input_shape))
	model.add(Dropout(0.3))
	model.add(Dense(300, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	return model

def count_word(tokens, lexicon_pos, lexicon_neg):
	word_count_pos = 0
	word_count_neg = 0
	for token in tokens:
		if token in lexicon_pos:
			word_count_pos += 1
		elif token in lexicon_neg:
			word_count_neg += 1
	return word_count_pos, word_count_neg

if __name__ == '__main__':

	# load np data
	files_pos = listdir('review_polarity/pos/')
	files_neg = listdir('review_polarity/neg/')

	X_corp = []
	X_embed = []
	y = np.concatenate((np.ones(1000).astype(int), np.zeros(1000).astype(int)))

	for file_path in files_pos:
		f_corp = open('review_polarity/pos/'+file_path, 'r')
		corp = f_corp.read().replace('\n','')
		f_corp.close()
		X_corp.append(corp)

		embeds = np.load('review_polarity/pos_np/' + file_path.replace('txt','npy'))
		embed_avg = embeds.mean(axis=0)
		X_embed.append(embed_avg)
		

	for file_path in files_neg:
		f_corp = open('review_polarity/neg/'+file_path, 'r')
		corp = f_corp.read().replace('\n','')
		f_corp.close()
		X_corp.append(corp)

		embeds = np.load('review_polarity/neg_np/' + file_path.replace('txt','npy'))
		embed_avg = embeds.mean(axis=0)
		X_embed.append(embed_avg)

	X_corp = np.stack(X_corp)
	X_embed = np.stack(X_embed)
	X_corp, X_embed, y = unison_shuffle(X_corp, X_embed, y)

	# load lexicon
	lexicon_pos = file_read_lexicon('opinion-lexicon-English/positive-words.txt')
	lexicon_neg = file_read_lexicon('opinion-lexicon-English/negative-words.txt')

	# cross val
	k = 10
	diff_sum = 0
	for i in range(k):
		X_embed_test = X_embed[int(i/k*1000):int((i+1)/k*1000)]
		X_corp_test = X_corp[int(i/k*1000):int((i+1)/k*1000)]
		y_test = y[int(i/k*1000):int((i+1)/k*1000)]

		X_embed_train = np.concatenate((X_embed[:int(max((i-1)/k*1000, 0))], X_embed[int((i+1)/k*1000):]))
		y_train = np.concatenate((y[:int(max((i-1)/k*1000, 0))], y[int((i+1)/k*1000):]))

		# MLP train
		model = model_architecture(input_shape=(300,))
		model.compile(loss='categorical_crossentropy', optimizer='Adagrad',metrics=['accuracy'])
		model.fit(X_embed_train, to_categorical(y_train), epochs=30, batch_size=100, verbose=0, validation_split=0.1)
		# MLP predict
		y_pred_test_mlp = np.argmax(model.predict(X_embed_test),axis=1)

		# word-count predict
		y_pred_test_wc = []
		for corp in X_corp_test:
			tokens = word_tokenize(corp)
			word_count_pos, word_count_neg = count_word(tokens, lexicon_pos, lexicon_neg)
			if word_count_pos > word_count_neg:
				y_pred_test_wc.append(1)
			else:
				y_pred_test_wc.append(0)
		y_pred_test_wc = np.array(y_pred_test_wc)

		# accuracy
		accuracy_mlp = sum(y_pred_test_mlp == y_test)/(1000/n)
		accuracy_wc = sum(y_pred_test_wc == y_test)/(1000/n)

		print('accuracy_mlp: ', accuracy_mlp, ' accuracy_wc: ', accuracy_wc)
