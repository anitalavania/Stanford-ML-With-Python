# gradientDescent() >

import numpy as np
from computeCost import *

def gradientDescent(X, y, theta, alpha, num_iter):
	m = len(y)
	J_history = np.zeros([num_iter, 1])
	for i in range(0, num_iter):
		#print('theta: ',theta,'\n')
		#print('alpha/m: ',alpha/m,'\n')
		#part1 = theta - (alpha/m) * (X.transpose().dot(X.dot(theta) -y))		
		#print('part1: ',part1,'\n')
		theta = theta - (alpha/m) * X.transpose().dot(X.dot(theta)-y) 
		#print('theta: ',theta,'\n')
		J_history[i, 0] = computeCost(X, y, theta)

	return theta, J_history
