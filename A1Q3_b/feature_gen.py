# Question 3 (b)
# Construct bow vectors

import pandas as pd
import pickle
import argparse

def bow_gen(data, vocab):
	bow_dict = {i:[0]*len(data) for i in vocab}
	bow = pd.DataFrame(data=bow_dict, index=[i for i in range(len(data))])
	for i in range(len(data)):
		for word in data['tokens'].iloc[i]:
			if word in vocab:
				bow.at[i,word] += 1
	return bow

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", type=int, default=2, help="vocab size 1e(?)")
	args = parser.parse_args()

	# load data
	train = pd.read_pickle('train.pkl').reset_index(drop=True)
	dev = pd.read_pickle('dev.pkl').reset_index(drop=True)
	test = pd.read_pickle('test.pkl').reset_index(drop=True)

	vocab_path = 'vocab_e'+str(args.v)+'.txt'
	with open(vocab_path, "rb") as fp:
		vocab = pickle.load(fp)
	
	# construct bag-of-words vector
	bow_train = bow_gen(train, vocab)
	bow_dev = bow_gen(dev, vocab)
	bow_test = bow_gen(test, vocab)

	bow_train.to_csv('bow_train_e'+str(args.v)+'.csv')
	bow_dev.to_csv('bow_dev_e'+str(args.v)+'.csv')
	bow_test.to_csv('bow_test_e'+str(args.v)+'.csv')

	# construct y
	#train['majortopic'].to_csv('y_train.csv')

	
	
