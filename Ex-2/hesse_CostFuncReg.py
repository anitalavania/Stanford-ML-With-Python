# hesse_CostFuncReg.py >

import numpy as np
from sigmoid import *

def hesse_CostFuncReg(theta, X, y, Lambda):
	(m,np1) = X.shape

	H = np.zeros([np1,np1])

	nume = np.exp(-1.*(X.dot(theta)))
	deno = np.square(1 + nume)
	S = nume/deno

	for j in range(0, np1):
		for k in range(0, np1):
			Xj = X[:,j]
			Xk = X[:,k]
			H[j,k] = sum(Xj*S*Xk)	

	print('H.shape: ',H.shape,'\n')

	H = H + Lambda*(np.identity(np1))
	H = H/m	

	return H
