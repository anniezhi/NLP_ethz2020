# To split randomly 200 pos and 200 neg as test set
import random
import shutil
import os

idx_list_pos = random.sample(range(1000),200)
idx_list_neg = random.sample(range(1000),200)

filename_list_pos = []
filename_list_neg = []

for idx in idx_list_pos:
	if idx < 10:
		filename_list_pos.append('cv00' + str(idx)) 
	elif idx >= 10 and idx < 100:
		filename_list_pos.append('cv0' + str(idx))
	else:
		filename_list_pos.append('cv' + str(idx))
filenames_all_pos = os.listdir('pos/')
for filename in filenames_all_pos:
	cvnum = filename.split('_')[0]
	if cvnum in filename_list_pos:
		shutil.move('pos/'+filename,'pos_test/')

for idx in idx_list_neg:
	if idx < 10:
		filename_list_neg.append('cv00' + str(idx))
	elif idx >= 10 and idx < 100:
		filename_list_neg.append('cv0' + str(idx))
	else:
		filename_list_neg.append('cv' + str(idx))
filenames_all_neg = os.listdir('neg/')
for filename in filenames_all_neg:
	cvnum = filename.split('_')[0]
	if cvnum in filename_list_neg:
		shutil.move('neg/'+filename,'neg_test/')

