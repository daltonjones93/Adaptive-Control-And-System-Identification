import numpy as np
import matplotlib.pyplot as plt



## estimate least squares for vector and quadratic



def least_squares_estimator(X,Y, eps = .001):
	try:
		inv =np.linalg.inv(X.T.dot(X))
	except:
		print('X does not have full rank')
		inv = np.linalg.inv(X.T.dot(X)+eps*np.eye(X.shape[1]))

	return inv.dot(X.T.dot(Y))

def quadratic_estimator_Q(X,r,eps = .001):
	Xmat = np.zeros((X.shape[0],X.shape[1]**2))
	for i in range(X.shape[0]):
		Xmat[i,:] = np.outer(X[i,:],X[i,:]).flatten()

	inv = np.linalg.inv(Xmat.T.dot(Xmat)+eps*np.eye(Xmat.shape[1]))
	Qhat = inv.dot(Xmat.T.dot(r))
	Q =Qhat.reshape((X.shape[1],X.shape[1]))
	return Q


def quadratic_estimator_Q_R(X,U,r, eps = .001):
	
	try:
		U.shape[1]
	except:
		U = U.reshape((U.shape[0],1))



	assert X.shape[0] == U.shape[0], 'X and U do not have the same number of data points'
	
	Xmat = np.zeros((X.shape[0],(X.shape[1]+U.shape[1])**2))

	for i in range(X.shape[0]):
		x = np.zeros(Xmat.shape[1])
		xsmall = np.zeros(X.shape[1]+U.shape[1])
		xsmall[:X.shape[1]] = X[i,:]
		xsmall[X.shape[1]:] = U[i,:]
		x = np.outer(xsmall,xsmall).flatten()
		Xmat[i,:] = x

		# Xmat[i,:X.shape[1]**2] = np.outer(X[i,:],X[i,:]).flatten()
		# Xmat[i,X.shape[1]**2:] = np.outer(U[i,:],U[i,:]).flatten()

	inv = np.linalg.inv(Xmat.T.dot(Xmat)+eps*np.eye(Xmat.shape[1]))

	QRhat = inv.dot(Xmat.T.dot(r))

	
	QR = QRhat.reshape((X.shape[1]+U.shape[1],X.shape[1]+U.shape[1]))

	Q = QR[:X.shape[1],:X.shape[1]]
	R = QR[X.shape[1]:,X.shape[1]:]
	return Q,R


#we need to evaluate the effectiveness of these estimators:

if __name__ == '__main__':

	np.random.seed(10)


	X = np.random.normal(0,1,(200,3))
	U = np.random.normal(0,1,200)
	Q = np.random.rand(3,3)
	Q = Q.T.dot(Q)

	print(U.shape)

	R = np.random.rand(2,2)
	R = R.T.dot(R)

	R = np.array([[1]])

	r_Q = []

	r_Q_R = []

	for i in range(200):
		x = X[i,:]
		u = U[i]
		r_Q.append(x.dot(Q.dot(x)))
		r_Q_R.append(x.dot(Q.dot(x))+(u**2)*R)


	r_Q = np.array(r_Q)
	r_Q_R = np.array(r_Q_R).reshape((200,))

	print(r_Q_R.shape)

	Qhat1 = quadratic_estimator_Q(X,r_Q)
	Qhat2,Rhat2 = quadratic_estimator_Q_R(X,U,r_Q_R)

	print(Q-Qhat1)
	print(Q-Qhat2)
	print(R-Rhat2)










