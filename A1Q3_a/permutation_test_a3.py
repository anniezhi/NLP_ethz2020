# Question 3 (a)(iii)
# http://axon.cs.byu.edu/Dan/478/assignments/permutation_test.php

import numpy as np

if __name__ == '__main__':
	k = 10
	accuracy_mlp = np.array([0.77, 0.78, 0.79, 0.76, 0.61, 0.77, 0.7, 0.75, 0.63, 0.73])
	accuracy_wc = np.array([0.7, 0.61, 0.68, 0.67, 0.83, 0.73, 0.67, 0.72, 0.69, 0.71])
	diff_avg = np.sum(accuracy_mlp - accuracy_wc) / k
	print(diff_avg)

	n = 2**k
	count_n = 0
	for i in range(n):
		sign = np.random.randint(2,size=k)
		sign = np.where(sign==0, -1, sign)
		acc_temp_mlp = accuracy_mlp * sign
		acc_temp_wc = accuracy_wc * sign
		diff_avg_new = np.sum(acc_temp_mlp - acc_temp_wc) / k
		if abs(diff_avg_new) >= abs(diff_avg):
			count_n += 1

	print(count_n)


