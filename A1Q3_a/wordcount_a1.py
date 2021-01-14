# Question 3 (a)(i)
# Tokenize the test set; Count neg/pos words; Classify by word count

from nltk.tokenize import word_tokenize
from os import listdir

def file_read_corpora(file_path):
	#input type: str
	#output type: str
	f = open(file_path, 'r')
	file_content = f.read()
	file_content = file_content.replace('\n','')
	f.close()
	return file_content

def file_read_lexicon(file_path):
	#input type: str
	#output type: list
	f = open(file_path, 'r', encoding='ISO-8859-1')
	file_content = f.read()
	lexicon_list = file_content.split('\n')
	lexicon_list.pop()
	return lexicon_list

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
	lexicon_pos = file_read_lexicon('opinion-lexicon-English/positive-words.txt')
	lexicon_neg = file_read_lexicon('opinion-lexicon-English/negative-words.txt')

	files_pos = listdir('review_polarity/pos_test/')
	files_neg = listdir('review_polarity/neg_test/')

	TP, FN, FP, TN = 0, 0, 0, 0

	for file_path in files_pos:
		file_content = file_read_corpora('review_polarity/pos_test/' + file_path)
		tokens = word_tokenize(file_content)
		word_count_pos, word_count_neg = count_word(tokens, lexicon_pos, lexicon_neg)
		
		if word_count_pos > word_count_neg:   #truth Pos, label Pos
			TP += 1
		else:								  #truth Pos, label Neg
			FN += 1

	for file_path in files_neg:
		file_content = file_read_corpora('review_polarity/neg_test/' + file_path)
		tokens = word_tokenize(file_content)
		word_count_pos, word_count_neg = count_word(tokens, lexicon_pos, lexicon_neg)

		if word_count_pos > word_count_neg:	  #truth Neg, label Pos
			FP += 1
		else:								  #truth Neg, label Neg
			TN += 1

	print('TP: ', TP, '\n')
	print('FN: ', FN, '\n')
	print('FP: ', FP, '\n')
	print('TN: ', TN)