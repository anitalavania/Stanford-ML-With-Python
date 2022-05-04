# computeCost() >
# Polynomial hypothesis

def computeCost(X, y, theta):
	m = len(y)
	res_mat = X.dot(theta) - y
	res_mat_sqr = (res_mat.transpose()).dot(res_mat)
	J = res_mat_sqr/(2.0*m)

	return J

	#print(J)
	#print("\n\n")
