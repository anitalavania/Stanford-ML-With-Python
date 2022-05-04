# CostFunc() >
# Sigmoid hypothesis: Lies between 0 and 1 as reqiured

import numpy as np
from sigmoid import *

def CostFunc(theta, X, y):
#def CostFunc(theta):


	#filename = '/home/anita/Learning/Machine_learning/Take-2/machine-learning-ex2/ex2/ex2data1.txt'
	#data = np.loadtxt(filename, delimiter=',', dtype='float')
	#print(data[0:10,:])
	#X = np.c_[data[:,:2]]
	#y = np.c_[data[:,2]]
	#from plotData import *
	#plotData(X, y)
	#(m,n) = X.shape
	#X = np.column_stack((np.ones(m), X))

	print('shape of theta in CostFunc: ',theta.shape,'\n')

	(m,n) = X.shape

	pos_J = y.transpose().dot(np.log(sigmoid(X.dot(theta))))			#(1 x m)*log[(m x (n+1)) * ((n+1) x 1)]
	neg_J = (1-y).transpose().dot(np.log(1 - sigmoid(X.dot(theta))))

	J = (-1/m)*(pos_J + neg_J)

	return J
