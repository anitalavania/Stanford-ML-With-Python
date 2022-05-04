# grad_CostFunc() >
#EDIT(04/05/2022): You do not need to convert these 1D arrays into 2D arrays. You can very well work with them. Just use flatten() function for arrays that get called as 2D arrays (such as 'y') whenever
#                  necessary, for e.g. while adding or subtracting.   

import numpy as np
from sigmoid import *

def grad_CostFunc(ini_theta, X, y):

	#print('Shape of ini_theta in grad_CostFunc: ',ini_theta.shape,'\n')	
	(m,np1) = X.shape
	
	#theta = np.array([ini_theta])
	#theta = theta.transpose()
	#print('shape of theta (after array-tion): ',theta.shape,'\n')

	z = X.dot(ini_theta)				# (mx(n+1)) * ((n+1)x1)
	grad = (X.transpose()).dot(sigmoid(z) - y.flatten())	# ((n+1)xm) * (mx1)
	grad = grad/m
	#fin_grad = grad.flatten()


	return grad
