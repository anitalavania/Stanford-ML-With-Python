#ex1.py >

print("Running warmUpExercise... \n")
print("5x5 Identity matrix: \n")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n=5

from warmUpExercise import *

warmUpExercise(n)

print("Loading and plotting data from ex1data1.txt... \n")
filename = 'ex1data1.txt'
data = np.loadtxt(filename, delimiter=',', dtype='float') #, skiprows=90) #, dtype='float') 
print(data)

#X = data[:, 0]
#y = data[:, 1]
X = np.c_[data[:, 0]]
y = np.c_[data[:, 1]]

m = len(y)	#number of training examples

#print(X)

from plotData import *
plotData(X,y)

####################### Cost function and GD #########################

X = np.c_[np.ones(m), data[:, 0]]  # [np.ones([m,1]), data[:,1]]
theta = np.zeros([2,1])

from computeCost import *
computeCost(X, y, theta)
computeCost(X, y, [[-1], [2]])

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;


from gradientDescent import *
theta = gradientDescent(X, y, theta, alpha, iterations)

print(theta)

######################################################################
######################## plot and predict ############################
######################################################################

plt.plot(X[:, 1], X.dot(theta))
plt.show(block=False)

x1 =  np.array([1, 3.5])
x2 =  np.array([1, 7.0])

prediction1 = x1.dot(theta) #35000 population
prediction2 = x2.dot(theta) #70000 population

pred1 = prediction1*10000.0
pred2 = prediction2*10000.0

print("Predicted revenue for a population of 35000 is", pred1, "\n") 
print("Predicted revenue for a population of 70000 is", pred2)

######################################################################
############################# J - plots ##############################
######################################################################

theta0_vals = np.array([-10,-5,-4,-3.8,-3.6,-3.4,-3.2,-3.0,-2.8,-2.6,-2.4,-2.2,-2.0,-1,0,1,2,3,4,5,6,7,8,9,10])
theta1_vals = np.array([-1,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.6,1.8,2.0,2.4,2.8,3.2,3.6,4.0])

print(theta0_vals[0])

J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

for i in range(0, len(theta0_vals)):
	for j in range(0, len(theta1_vals)):
		t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
		J_vals[i, j] = computeCost(X, y, t)
		#print(J_vals[i,j]) 
		

J_vals = J_vals.transpose()

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

#fig = plt.figure()
#axes = fig.gca(projection = '3d')
plt.figure()
axes = plt.gca(projection = '3d')
axes.plot_surface(theta0_vals, theta1_vals, J_vals)
plt.xlabel("$\u03F4_{0}$")
plt.ylabel("$\u03F4_{1}$")
axes.set_zlabel("J")

#fig, axes = plt.subplots(1, 1)
#axes.contour(theta0_vals, theta1_vals, J_vals, 100)
#plt.figure()
#plt.contour(theta0_vals, theta1_vals, J_vals, 100)

plt.show()
