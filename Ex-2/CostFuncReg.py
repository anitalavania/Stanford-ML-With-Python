# CostFuncReg() >
# Sigmoid hypothesis: Lies between 0 and 1 as reqiured

import numpy as np
from sigmoid import *

def CostFuncReg(theta, X, y, Lambda):

	J=0
	(m,n) = X.shape
	#reg_theta = theta
	#reg_theta[0] = 0

	#or...
	reg_theta = np.append(0, theta[1:])
	print('Shape of reg_theta in CostFuncReg: ',reg_theta.shape,'\n\n')

	pos_J = y.transpose().dot(np.log(sigmoid(X.dot(theta))))	#(1 x m)*log[(m x (n+1)) * ((n+1) x 1)]
	neg_J = (1-y).transpose().dot(np.log(1 - sigmoid(X.dot(theta))))

	J = (-1/m)*(pos_J + neg_J) + (Lambda*reg_theta.transpose().dot(reg_theta))/(2*m)
	#J = (-1/m)*(pos_J + neg_J) + (Lambda*sum(reg_theta**2))/2*m

	#grad = ((X.transpose()).dot(sigmoid(X.dot(theta)) - y.flatten()) + Lambda*reg_theta)/m

	#return J, grad
	return J
