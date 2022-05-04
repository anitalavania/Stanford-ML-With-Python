# plotData.py

import matplotlib.pyplot as plt_dp

def plotData(X,y):
	plt_dp.figure()
	plt_dp.scatter(X, y, marker='x', color='red')
	plt_dp.xlabel('Population')
	plt_dp.ylabel('Revenue')

	plt_dp.show(block=False)	
	#plt_dp.show()

