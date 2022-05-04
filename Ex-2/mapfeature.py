#mapfeature

import numpy as np

def mapfeature(X):
	degree = 6
	(m,n) = X.shape
	X_new = np.ones([m,1])
	X1 = np.c_[X[:,0]]
	X2 = np.c_[X[:,1]]
	#print('X1: ',X1)
	for i in range(1,degree+1):
		for j in range(0,i+1):
			X_new = np.column_stack((X_new, X1**(i-j)*X2**j))		#28 columns now: (m x 28)

	return X_new


