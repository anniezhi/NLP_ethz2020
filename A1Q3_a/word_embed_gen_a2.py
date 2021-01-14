# Question 3 (a)(ii) - preprocessing
# Tokenize the texts; Word embeddings from GloVe

from nltk.tokenize import word_tokenize
from os import listdir
import numpy as np

def file_read_corpora(file_path):
	#input type: str
	#output type: str
	f = open(file_path, 'r')
	file_content = f.read()
	file_content = file_content.replace('\n','')
	f.close()
	return file_content

if __name__ == '__main__':
	embeddings_dict = {}
	with open('glove/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
		for line in f:
			values = line.split()
			word = values[0]
			vector = np.asarray(values[1:],'float32')
			embeddings_dict[word] = vector

	files_pos_train = listdir('review_polarity/pos_train/')
	files_neg_train = listdir('review_polarity/neg_train/')
	files_pos_test = listdir('review_polarity/pos_test/')
	files_neg_test = listdir('review_polarity/neg_test/')

	for file_path in files_pos_train:
		file_content = file_read_corpora('review_polarity/pos_train/' + file_path)
		tokens = word_tokenize(file_content)
		embeds = []
		for token in tokens:
			if token in embeddings_dict:
				embed = embeddings_dict[token]
				embeds.append(embed)
		embeds = np.stack(embeds)
		np.save('review_polarity/pos_train_np/'+file_path.replace('txt','npy'), embeds)

	for file_path in files_neg_train:
		file_content = file_read_corpora('review_polarity/neg_train/' + file_path)
		tokens = word_tokenize(file_content)
		embeds = []
		for token in tokens:
			if token in embeddings_dict:
				embed = embeddings_dict[token]
				embeds.append(embed)
		embeds = np.stack(embeds)
		np.save('review_polarity/neg_train_np/'+file_path.replace('txt','npy'), embeds)

	for file_path in files_pos_test:
		file_content = file_read_corpora('review_polarity/pos_test/' + file_path)
		tokens = word_tokenize(file_content)
		embeds = []
		for token in tokens:
			if token in embeddings_dict:
				embed = embeddings_dict[token]
				embeds.append(embed)
		embeds = np.stack(embeds)
		np.save('review_polarity/pos_test_np/'+file_path.replace('txt','npy'), embeds)

	for file_path in files_neg_test:
		file_content = file_read_corpora('review_polarity/neg_test/' + file_path)
		tokens = word_tokenize(file_content)
		embeds = []
		for token in tokens:
			if token in embeddings_dict:
				embed = embeddings_dict[token]
				embeds.append(embed)
		embeds = np.stack(embeds)
		np.save('review_polarity/neg_test_np/'+file_path.replace('txt','npy'), embeds)