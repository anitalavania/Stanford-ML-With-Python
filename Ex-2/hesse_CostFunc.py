# hesse_CostFunc() >
# Hesse matrix: H_{jk} = dJ(theta)/([d theta_j] [d theta_k])

import numpy as np
from sigmoid import *

def hesse_CostFunc(ini_theta, X, y):
	
	(m,np1) = X.shape					#np1 is (n+1)
	n = np1-1

	H = np.zeros([n+1, n+1])

	nume = np.exp(-1.* X.dot(ini_theta))			#nume.shape: (m,)
	deno = 1. + nume					#deno.shape: (m,)
	S = nume/np.square(deno)				#S.shape: (m,)

	for j in range(0, n+1):
		for k in range(0, n+1):
			#Xk = np.c_[X[:,k]]			#Xk.shape: (m,1)
			#Xj = np.c_[X[:,j]]			#Xj.shape: (m,1)
			#H[j,k] = sum(Xk*S*Xj)			
			Xk = X[:,k]				#Xk.shape: (m,)
			Xj = X[:,j]				#Xj.shape: (m,)
			H[j,k] = sum(Xk*S*Xj)

			#for i in range(0, m):
			#	Xi = np.array([X[i,:]])
			#	H[j,k] += X[i,k]*X[i,j]*( np.exp(-1. * Xi.dot(theta))/(1. + np.square(np.exp(-1. * Xi.dot(theta)))) )


	H =  H/m

	return H
