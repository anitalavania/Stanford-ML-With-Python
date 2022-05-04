# featureNormalize() >

import numpy as np

def featureNormalize(X_norm):
	mu = np.zeros([1, len(X_norm[0])])
	sigma = np.zeros([1, len(X_norm[0])])


	mu = X_norm.mean(0)
	sigma = X_norm.std(0)

	X_norm =  (X_norm-mu)/sigma

	return X_norm, mu, sigma
