# Question 3 (a)(ii)
# Load word embeddings; MLP (2 hidden layers); softmax (Neg/Pos)

import numpy as np
from os import listdir
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import h5py

def unison_shuffle(a,b):
	assert(len(a)==len(b))
	p = np.random.permutation(len(a))
	return a[p], b[p]

def model_architecture(input_shape):
	model = Sequential()
	model.add(Dense(300, activation='relu', input_shape=input_shape))
	model.add(Dropout(0.3))
	model.add(Dense(300, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	return model

if __name__ == '__main__':

	# load data
	files_pos_train = listdir('review_polarity/pos_train_np/')
	files_neg_train = listdir('review_polarity/neg_train_np/')
	files_pos_test = listdir('review_polarity/pos_test_np/')
	files_neg_test = listdir('review_polarity/neg_test_np/')

	X_train = []
	y_train = np.concatenate((np.ones(800).astype(int), np.zeros(800).astype(int)))

	for file_path in files_pos_train:
		embeds = np.load('review_polarity/pos_train_np/' + file_path)
		embed_avg = embeds.mean(axis=0)
		X_train.append(embed_avg)

	for file_path in files_neg_train:
		embeds = np.load('review_polarity/neg_train_np/' + file_path)
		embed_avg = embeds.mean(axis=0)
		X_train.append(embed_avg)

	X_train = np.stack(X_train)
	X_train, y_train = unison_shuffle(X_train, y_train)
	y_train = to_categorical(y_train)

	X_test = []
	y_test = np.concatenate((np.ones(200).astype(int), np.zeros(200).astype(int)))
	y_test = to_categorical(y_test)

	for file_path in files_pos_test:
		embeds = np.load('review_polarity/pos_test_np/' + file_path)
		embed_avg = embeds.mean(axis=0)
		X_test.append(embed_avg)

	for file_path in files_neg_test:
		embeds = np.load('review_polarity/neg_test_np/' + file_path)
		embed_avg = embeds.mean(axis=0)
		X_test.append(embed_avg)

	X_test = np.stack(X_test)

	# MLP model
	model = model_architecture(input_shape=(300,))
	model.compile(loss='categorical_crossentropy', optimizer='Adagrad',metrics=['accuracy'])
	#es = EarlyStopping(monitor='val_acc', mode='max',verbose=1, baseline=0.8, patience=3)
	model.fit(X_train, y_train, epochs=35, batch_size=100, verbose=1, validation_split=0.1)
	model.save('mlp_a2.h5')

	# predict
	y_pred_test = np.argmax(model.predict(X_test),axis=1)
	print(y_pred_test)

	# analysis
	y_pred_test_pos = y_pred_test[:200]
	y_pred_test_neg = y_pred_test[200:]
	TP = sum(y_pred_test_pos==1)
	FN = sum(y_pred_test_pos==0)
	FP = sum(y_pred_test_neg==1)
	TN = sum(y_pred_test_neg==0)
	print('TP = ',TP,' | FN = ',FN,' | FP = ',FP,' | TN = ',TN)