# Question 3 (b)
# To split randomly into 60% training, 20% dev, 20% test set
# 31034 samples in total

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string

if __name__ == '__main__':
	# read data
	df = pd.read_csv(r'boydstun_nyt_frontpage_dataset_1996-2006_0_pap2014_recoding_updated2018.csv', encoding='latin-1')
	# fill na
	df['title'] = df['title'].fillna('')
	df['summary'] = df['summary'].fillna('')

	# tokenization
	df['strings'] = df['title'] + ' ' + df['summary']
	df['tokens'] = [None] * len(df)
	porter = PorterStemmer()
	stop_words = set(stopwords.words('english'))
	for i in range(len(df)):
		strings = df['strings'][i].lower()
		#stemming
		stem_string = []
		for word in strings.split():
			stem_string.append(porter.stem(word) + ' ')
		stem_string = "".join(stem_string) #list -> str
		#tokenization
		tokens = word_tokenize(stem_string)
		#stop word removal
		filtered_tokens = [w for w in tokens if w not in stop_words and w not in string.punctuation]
		#save
		df.at[i,'tokens'] = filtered_tokens
		#print(type(filtered_tokens))

	# train, dev, test split
	train,test = train_test_split(df, test_size=0.4)
	dev, test = train_test_split(test, test_size=0.5)

	train.to_pickle('train.pkl')
	dev.to_pickle('dev.pkl')
	test.to_pickle('test.pkl')

	word_freq = train.tokens.explode().value_counts()    #type: pd.series

	vocab_e2 = word_freq.index[:int(1e2)].tolist()   #type: list
	vocab_e3 = word_freq.index[:int(1e3)].tolist()
	vocab_e4 = word_freq.index[:int(1e4)].tolist()
	vocab_e5 = word_freq.index[:int(1e5)].tolist()

	# save vocab
	with open("vocab_e2.txt", "wb") as fp:   #Pickling
		pickle.dump(vocab_e2, fp)
	with open("vocab_e3.txt", "wb") as fp:
		pickle.dump(vocab_e3, fp)
	with open("vocab_e4.txt", "wb") as fp:
		pickle.dump(vocab_e4, fp)
	with open("vocab_e5.txt", "wb") as fp:
		pickle.dump(vocab_e5, fp)