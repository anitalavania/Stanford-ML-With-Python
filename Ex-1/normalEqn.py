# normalEqn >

import numpy as np

def normalEqn(X, y):
	theta = (np.linalg.inv(X.transpose().dot(X))).dot(X.transpose().dot(y))

	return theta
