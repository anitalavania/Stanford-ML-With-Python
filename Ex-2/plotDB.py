# plotDB() >

import numpy as np
import matplotlib.pyplot as plt
from  plotData import *
from mapfeature import *

def plotDB(theta, X, y):
	#pos_X = np.array([[0, 0]])	
	#neg_X = np.array([[0, 0]])	
	#(m, np1) = X.shape
	#for i in range(0, m):
	#	if(y[i]==1):
	#		pos_X.vstack([pos_X, X[i,1:2]])
	#	else :
	#		neg_X.vstack([neg_X, X[i,1:2]])

	print('shape of X = ',X.shape,'\n')

	X_prime = np.array(X[:,1:3]);
	print('shape of X_prime = ',X_prime.shape,'\n')

	if(X.shape[1] <= 3):
		x_db = np.arange(min(X[:,1])-2, max(X[:,2])+2)
		y_db = (-1/theta[2]) * (theta[0] + theta[1]*x_db)
		plt.plot(x_db, y_db);
		plt.show()

	else:
		u = np.linspace(-1,1.5,50)
		v = np.linspace(-1,1.5,50)
		z = np.zeros([u.shape[0], v.shape[0]]) 
		for i in range(len(u)):
			for j in range(len(v)):
				X_mf = np.array([[u[i], v[j]]])
				z[i,j] = mapfeature(X_mf).dot(theta)

		print('shape of z: ',z.shape,'\n')
		z = z.transpose()

		fig, ax = plt.subplots(1,1)

		plotData(X_prime,y, plt)
		ax.contour(u,v,z)
		plt.show(block=False)

		
		

