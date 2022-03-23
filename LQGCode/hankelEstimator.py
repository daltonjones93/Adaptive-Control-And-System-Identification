

### hankel estimation code
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


def create_orthogonal_matrix(dim):
    A = np.random.rand(dim,dim)
    A[0,:] /= np.linalg.norm(A[0,:])
    for i in range(1,dim):
        for j in range(i):
            A[i,:] -= A[j,:]*(A[i,:].dot(A[j,:]))
        A[i,:] /= np.linalg.norm(A[i,:])
    
    return A




class HankelEstimator(object):
	def __init__(self,u,y,state_space_dim,hankel_dim):

		
		self.u = u
		self.y = y

		assert len(u) == len(y), "make sure the output and input have the same length and do not include y[0]"
		self.n = state_space_dim
		self.hankel_dim = hankel_dim

	def createHankelMatrix(self):

		#create the u matrix of inputs 
		#(what should the dimesion of this be)
		Umat = np.zeros((len(self.u) - 2*self.hankel_dim,self.hankel_dim))
		Ymat = np.zeros(Umat.shape)
		for i in range(Umat.shape[0]):
			Umat[i,:]=np.flip(self.u[i:i+self.hankel_dim])
			Ymat[i,:]=self.y[i+self.hankel_dim-1:i+2*self.hankel_dim-1]

		hankel_mat = np.linalg.inv(Umat.T.dot(Umat))
		hankel_mat = hankel_mat.dot(Umat.T.dot(Ymat))

		self.hankel_mat = hankel_mat

	def estimateSystemParameters(self):
		
		self.createHankelMatrix()

		u,sigma,vh = np.linalg.svd(self.hankel_mat)

		k = self.n

		sigma_red = np.sqrt(np.diag(sigma))[:k,:k]

		ured = u[:,:k].dot(sigma_red)
		vred = sigma_red.dot(vh[:k,:])
		self.Chat = ured[0,:]
		self.Bhat = vred[:,0]


		umin = ured[:-1,:]
		uplus = ured[1:,:]

		uinv = np.linalg.inv(umin.T.dot(umin))

		self.Ahat = uinv.dot(umin.T.dot(uplus))

		eigs= np.linalg.eigvals(self.Ahat)

		if np.max(np.abs(eigs)) >1:
			print('Reducing estimate of A to make system stable')
			w,vl,vr = scipy.linalg.eig(self.Ahat,left = True,right = True)
			for i in range(len(w)):
				if np.abs(w[i]) >1:
					w[i] = np.sign(w[i])*.999
					#use .99 to make this as big as possible
			self.Ahat = np.real(vr.dot(np.diag(w).dot(vl.T)))

		# assert np.max(np.abs(eigs)) <1, 'Estimated Matrix is not stable'







#need to evaluate how well this is working

if __name__ == '__main__':
	
	V= create_orthogonal_matrix(10)
	d = np.random.rand(10)
	A = V.T.dot(np.diag(d).dot(V))

	B = np.random.rand(10)
	C = np.random.rand(10).reshape((1,10))

	u = np.random.normal(0,1,100000)
	y = [0]
	x = np.zeros(10)

	v = np.random.normal(0,.1,(100000,10))

	
	for i in range(100000):
		x = A.dot(x) + B*u[i] +v[i,:]
		y.append(C.dot(x))

	y = np.array(y).reshape((len(y),))

	hankel_size = 50

	state_space_dimension = 10

	hankel = HankelEstimator(u,y,state_space_dimension,hankel_size)

	hankel.estimateSystemParameters()



	#create hankel matrix 
	realHankel = np.zeros((10,10))
	for i in range(10):
		for j in range(10):
			realHankel[i,j] = C.dot(np.linalg.matrix_power(A,i+j).dot(B))

	plt.figure()
	plt.plot(realHankel.flatten())
	plt.plot(hankel.hankel_mat[:10,:10].flatten())
	# print(realHankel-hankel.hankel_mat[:10,:10])
	plt.show()
	



	#create trajectory

	xhat = np.zeros(10)

	yhat = [0]
	
	for i in range(10000):
		xhat = hankel.Ahat.dot(xhat) + hankel.Bhat*u[i] +v[i,:]
		yhat.append(hankel.Chat.dot(xhat))

	

	yhat = np.array(yhat)

	plt.figure()
	plt.plot(y[:100],label = 'orig')
	plt.plot(yhat[:100],label ='estimated')
	plt.legend()
	plt.show()













