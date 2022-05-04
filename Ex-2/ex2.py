#Logistic regression exercise

import numpy as np

filename = '/Users/anita/Learning/Machine_learning/Take-2/machine-learning-ex2/ex2/ex2data1.txt'
data = np.loadtxt(filename, delimiter=',', dtype='float')
#print(data[0:10,:])

#Selecting a one or more columns out of a matrix using "np.c_[:, columns_you_want]"
X = np.c_[data[:,:2]]
y = np.c_[data[:,2]]

#from plotData import *
#plotData(X, y)

(m,n) = X.shape

#Adding a coulmn of ones using "np.column_stack(column_to_append, original matrix)"
X = np.column_stack((np.ones(m), X))

#Initialize weights
initial_theta = np.zeros([n+1,1])
initial_theta[0,0] = 1.0

#Importing modules
from CostFunc import *
from grad_CostFunc import *
from hesse_CostFunc import *
from p_hesse_CostFunc import *

#cost, grad = CostFunc(initial_theta, X, y)
cost = CostFunc(initial_theta, X, y)
#print('Predicted cost = ', cost,'\n')
#print('Predicted grad = ', grad,'\n\n\n')

test_theta = np.array([[-24], [.2], [.2]])
#cost, grad = CostFunc(test_theta, X, y)
#cost = CostFunc(test_theta, X, y)
#print('Predicted cost = ', cost,'\n')
#print('Predicted grad = ', grad)

###############################################
########### Cost minimization here ############
###############################################

from scipy.optimize import minimize

print('2nd row of X: ',np.array([X[1,:]]),'\n')
print('theta: ',test_theta,'\n')
print('2nd row of X.theta: ',np.array([X[1,:]]).dot(test_theta),'\n')

print('shape of ini_theta in main function: ',initial_theta.shape,'\n')

args = (X, y)
#res = minimize(CostFunc, initial_theta, args = args, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
#res = minimize(CostFunc, initial_theta, args = args, method='BFGS', jac=grad_CostFunc, options={'disp': True})
res = minimize(CostFunc, initial_theta, args = args, method='trust-ncg', jac=grad_CostFunc, hess=hesse_CostFunc, options={'gtol': 1e-8, 'disp': True})
#res = minimize(CostFunc, initial_theta, args = args, method='Newton-CG', jac=grad_CostFunc, hessp=p_hesse_CostFunc, options={'xtol': 1e-8, 'disp': True})
print(res.x)

fin_theta = res.x
print('fin_theta: ',fin_theta,'\n')
################################################
############ Plot decision boundary ############
############ (Only for 2 features) ############
################################################

from plotDB import *

plotDB(fin_theta, X, y)

###############################################
########## Prediction and accuracy ############
###############################################

test_example = np.array([[1, 45, 85]])
fin_theta = (np.asarray([fin_theta])).transpose()

#Predicting probability of admission

from sigmoid import *
admit_prob = sigmoid(test_example.dot(fin_theta))
print('Test admit probability = ',admit_prob,'\n')

#Accuracy on train set

pred = np.round(sigmoid(X.dot(fin_theta)))              #mx(n+1) * (n+1)x1
#print('pred: ',pred,'\n')
accuracy = np.mean(pred==y) * 100
print('Accuracy = ',accuracy,'\n')
