#Logistic regression exercise

import numpy as np

filename = '/Users/anita/Learning/Machine_learning/Take-2/machine-learning-ex2/ex2/ex2data2.txt'
data = np.loadtxt(filename, delimiter=',', dtype='float')
#print(data[0:10,:])

#Selecting a one or more columns out of a matrix using "np.c_[:, columns_you_want]"
X = np.c_[data[:,:2]]
y = np.c_[data[:,2]]

#from plotData import *
#plotData(X, y)

(m,n) = X.shape
print('Initial shape of X: ',(m,n),'\n')

from mapfeature import *

X_new = mapfeature(X)
print('X_new: ',X_new,'\n')
print('shape of X_new: ',X_new.shape,'\n')
(m,n) = X_new.shape
print('shape of X after adding polynomial features: ',X_new.shape,'\n')

#Testing cost function with test theta and lambda

#ini_theta = np.zeros([n,1])
ini_theta = np.zeros_like((np.r_[[X_new[0,:]]]).transpose())
Lambda = 1
print('ini_theta: ',ini_theta,'\n')

from CostFuncReg import *

#cost, grad = CostFuncReg(ini_theta, X_new, y, Lambda)
cost = CostFuncReg(ini_theta, X_new, y, Lambda)

print('J = ',cost,'\n')
#print('grad (Initial 10 elements) = ',grad[:10,:],'\n')

# Now with test_theta = all ones and lambda=10

test_theta = np.ones_like(ini_theta) 
Lambda = 10

print('Lambda = 10 now... \n')
print('test_theta: ',test_theta,'\n\n')

#cost, grad = CostFuncReg(test_theta, X_new, y, Lambda)
cost = CostFuncReg(test_theta, X_new, y, Lambda)
print('J = ',cost,'\n')
#print('grad (Initial 10 elements) = ',grad[:10,:],'\n')

##================== Part 2 ====================##
##=============== Optimization =================##
##==============================================##

theta = np.zeros_like(ini_theta)
Lambda=1

from scipy.optimize import minimize
from grad_CostFuncReg import *
from hesse_CostFuncReg import *

args = (X_new, y, Lambda)

#res = minimize(CostFuncReg, theta, args = args, method='nelder-mead', options={'xatol': 1e-7, 'disp': True})
#res = minimize(CostFuncReg, theta, args = args, method='BFGS', jac=grad_CostFuncReg, options={'disp': True})
res = minimize(CostFuncReg, theta, args=args, method='trust-ncg', jac=grad_CostFuncReg, hess=hesse_CostFuncReg, options={'gtol': 1e-8, 'disp': True})
print(res.x)

fin_theta = res.x
print('shape of fin_theta: ', fin_theta.shape)

from plotDB import *
plotDB(fin_theta, X_new, y)

from sigmoid import *
pred = np.round(sigmoid(X_new.dot(fin_theta)))

print('shape of pred: ',pred.shape,'\n')

accuracy = np.mean(pred==(y.flatten())) * 100
print('Accuracy: ',accuracy,'\n')









