# sigmoid.py >

import numpy as np

def sigmoid(z):
	g = 1./(1 + np.exp(-1.*z))

	return g
