# Question 3 (b)
# Log linear model

import pandas as pd
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight


def model_architecture(input_shape):
	model = Sequential()
	model.add(Dense(100, activation='relu', input_shape=input_shape))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(60, activation='relu'))
	model.add(Dense(32, activation='softmax'))
	return model

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", type=int, default=2, help="vocab size 1e(?)")
	parser.add_argument("-regularization", type=str, default=None, help="penalty")
	parser.add_argument("-C", type=float, default=1.0, help="penalty strength")
	args = parser.parse_args()

	# define penalty
	if args.regularization == None:
		penalty = 'none'
	elif args.regularization == 'l2':
		penalty = 'l2'

	# load data
	bow_train = pd.read_csv('bow_train_e'+str(args.v)+'.csv', index_col=False).drop(['Unnamed: 0'],axis='columns')
	bow_dev = pd.read_csv('bow_dev_e'+str(args.v)+'.csv', index_col=False).drop(['Unnamed: 0'],axis='columns')
	bow_test = pd.read_csv('bow_test_e'+str(args.v)+'.csv', index_col=False).drop(['Unnamed: 0'],axis='columns')
	
	train_np = bow_train.to_numpy()
	dev_np = bow_dev.to_numpy()
	test_np = bow_test.to_numpy()

	# modify y for classification
	y_train = pd.read_csv('y_train.csv', index_col=False).drop(['Unnamed: 0'],axis='columns').to_numpy().ravel()
	y_dev = pd.read_csv('y_dev.csv', index_col=False).drop(['Unnamed: 0'],axis='columns').to_numpy().ravel()
	y_test = pd.read_csv('y_test.csv', index_col=False).drop(['Unnamed: 0'],axis='columns').to_numpy().ravel()


	y_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 10:8,\
			  12:9, 13:10, 14:11, 15:12, 16:13, 17:14, 18:15, 19:16, 20:17, 21:18,\
			  23:19, 24:20, 26:21, 27:22, 29:23, 30:24, 31:25, 99:26}

	y_train = [y_dict.get(i,i) for i in y_train]
	y_dev = [y_dict.get(i,i) for i in y_dev]
	y_test = [y_dict.get(i,i) for i in y_test]

	#y_train[y_train == 99] = 32
	#y_train = y_train - 1
	#y_dev[y_dev == 99] = 32
	#y_dev = y_dev - 1
	#y_test[y_test == 99] = 32
	#y_test = y_test - 1

	#y_train_onehot = to_categorical(y_train)
	#y_dev_onehot = to_categorical(y_dev)
	#y_test_onehot = to_categorical(y_test)
	
	weights = compute_class_weight('balanced', classes=range(27), y=y_train)
	classes = list(range(27))
	class_weight = dict(zip(classes, weights))

	# NN
	#model = model_architecture(input_shape=(10**(args.v),))
	#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#model.fit(train_np, y_train_onehot, epochs=30, batch_size=100, verbose=1, validation_data=(dev_np, y_dev_onehot))
	#model.save('ll_no_penalty.h5')
	
	#y_pred_test = np.argmax(model.predict(test_np),axis=1)
	#print(metrics.classification_report(y_test, y_pred_test, digits=3))

	# Logistic Regression
	#clf = LogisticRegression(class_weight=class_weight).fit(train_np, y_train)
	clf = LogisticRegression(max_iter=200, penalty=penalty, C=args.C).fit(train_np, y_train)
	y_train_pred = clf.predict(train_np)
	y_dev_pred = clf.predict(dev_np)
	y_test_pred = clf.predict(test_np)
	print('train result \n')
	print(metrics.classification_report(y_train, y_train_pred, digits=3))
	print('val result \n')
	print(metrics.classification_report(y_dev, y_dev_pred, digits=3))
	print('test result \n')
	print(metrics.classification_report(y_test, y_test_pred, digits=3))