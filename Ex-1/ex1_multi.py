
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys				#for importing modules from a different directory

print("Loading and plotting data from ex1data1.txt... \n")
filename = '/home/anita/Learning/Machine_learning/Take-2/machine-learning-ex1/ex1/ex1data2.txt'
data = np.loadtxt(filename, delimiter=',', dtype='float') #, skiprows=90) #, dtype='float') 
#print(data)

#X = data[:, [0,1]]
#y = data[:, 2]
X = np.c_[data[:, [0,1]]]
y = np.c_[data[:, 2]]

m = len(X)
print(len(X))       #number of training examples (number of rows)
print(len(X[0]))    #number of features (number of columns)


from featureNormalize import *
X, mu, sigma = featureNormalize(X)

#X = np.c_[np.ones(m), data[:, [0,1]]]
X = np.c_[np.ones(m), X]
n = len(X[0])

print(X[:10,:])

alpha = 0.01
num_iters = 400
theta = np.zeros([n,1])

from gradientDescent import *
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

######## plotting the cost function ##########

plt.figure()
plt.plot(np.arange(1,len(J_history)+1), J_history)
plt.xlabel('Iteration number')
plt.ylabel('Cost (J)')
plt.show(block=False)

print('theta: ',theta,'\n')

################################################
# Estimate the price of a 1650 sq-ft, 3 br house
################################################

price = 0
feat = np.array([[1650, 3]])
feat = (feat-mu)/sigma
feat = np.c_[[1], feat]
print('feat: ',feat,'\n')
price = feat.dot(theta)
print('Predicted price for 1650 sq-ft, 3br house = ',price,'\n')


###############################################
# Method of Normal equations ##################
###############################################

from normalEqn import *
theta = normalEqn(X, y)

print('theta computed using Normal equation: ',theta,'\n')

################################################
# Estimate the price of a 1650 sq-ft, 3 br house
################################################

price = 0
feat = np.array([[1650, 3]])
feat = (feat-mu)/sigma
feat = np.c_[[1], feat]
print('feat: ',feat,'\n')
price = feat.dot(theta)
print('Predicted price for 1650 sq-ft, 3br house (Normal equation) = ',price,'\n')







