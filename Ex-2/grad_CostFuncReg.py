#grad_CostFuncReg.py >

import numpy as np
from sigmoid import *

def grad_CostFuncReg(theta, X, y, Lambda):
	(m,np1) = X.shape

	z = X.dot(theta)
	grad_basic = (1/m)*X.transpose().dot((sigmoid(z) - y.flatten()))
	grad_reg = Lambda*theta/m
	grad = grad_basic + grad_reg

	return grad 
