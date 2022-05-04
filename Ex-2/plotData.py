#plotData.py >

import numpy as np
#import matplotlib.pyplot as plt

def plotData(X, y, plt):

	#pos_X = np.array([[0, 0]])
	#neg_X = np.array([[0, 0]])

	pos_X = np.zeros_like([X[0,:]])
	neg_X = np.zeros_like([X[0,:]])

	for i in range(0, len(y)):
		#print('y = ',y[i],'\n')
		if (y[i]==1):
			pos_X = np.vstack((pos_X, X[i,:]))
		else :
			neg_X = np.vstack((neg_X, X[i,:]))
			
	#plt.figure()
	plt.scatter(pos_X[1:,0], pos_X[1:,1], marker='o', color='blue')
	plt.scatter(neg_X[1:,0], neg_X[1:,1], marker='x', color='red')
	plt.xlabel('Exam-I score')
	plt.ylabel('Exam-II score')
	plt.legend(["Admitted", "Not admitted"])
	plt.show(block=False)

