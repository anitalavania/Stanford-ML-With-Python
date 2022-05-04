# p_hesse_CostFunc() >

import numpy as np
from sigmoid import *

def p_hesse_CostFunc(ini_theta, p, X, y):
	(m,np1) = X.shape
	n = np1-1
	theta = np.zeros([n+1,1])
	for i in range(0, n+1):
		theta[i,0] = ini_theta[i]

	H = np.zeros([n+1, n+1])
	Hp = np.zeros([n+1, 1])

	nume = np.exp(-1. * X.dot(theta))
	deno = 1. + nume
	S = nume/np.square(deno)

	for k in range(0, n+1):
		for j in range(0, n+1):
			Xj = np.c_[X[:,j]]
			Xk = np.c_[X[:,k]]
			Hp[j, 0] += sum(Xk*S*Xj)*p[k]  

	Hp =  Hp/m

	return Hp
