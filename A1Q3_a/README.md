## Steps
(i) Move all train files from downloaded polarity data to one folder /train, and all test files to one folder /test. Remove the header of the downloaded lexicon files, and only remain the words.\
(ii) Run ```train_test_sep.py``` under the parent folder of /train and /test.\
(iii) Download GloVe word vectors from http://nlp.stanford.edu/data/glove.6B.zip, to the same folder with the lexicons and polarity data.\
(iv) For sub-problem a(i), run ```wordcount_a1.py```.\
(v) For sub-problem a(ii), run ```word_embed_gen_a2.py``` to generate word embeddings of the corpus. Then run ```mlp_a2.py``` to train and test an MLP with the embeddings.\
(vi) For sub-problem a(iii), run ```cross_val_a3.py``` to perform cross validation on the MLP model. Log the performances of each loop. Then run ```permutation_test_a3.py``` to perform statistical comparison between wordcount model and MLP via paired permutation test. (c.f. http://axon.cs.byu.edu/Dan/478/assignments/permutation_test.php)

## Problem description:
Download the Pang and Lee movie review data.2 Hold out a randomly-selected 400
reviews as a test set. Download a sentiment lexicon, such as the one currently available
from Bing Liu.3\
(i) Tokenize the data using a library of your choosing4 and classify each document as positive if it has more positive sentiment words than negative sentiment words. Compute the accuracy and F-measure5 on detecting positive reviews on the test set, using this lexicon-based classifier.\
(ii) Then train a discriminative classifier (averaged perceptron or logistic regression) on the training set, and compute its accuracy and F-measure on the test set.\
(iii) Qualitatively compare the difference in accuracy between the two methods. Does the discriminative classifier do overwhelmingly better than guessing the mode? Describe a method for quantifying the significance of the difference between the two methods.
