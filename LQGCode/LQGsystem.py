

import numpy as np
import matplotlib.pyplot as plt


#LQG system simulator



class LQGsystem:
	
	def __init__(self,A,B,C,covW,covV,Q,R):
		self.A = A
		self.B = B
		self.C = C
		self.covW = covW
		self.covV = covV
		self.Q = Q
		self.R = R
		self.state=np.zeros(A.shape[0])

	def reset_state(self):
		self.state = np.zeros(self.A.shape[0])
		self.input = []
		self.output = []
		self.cost = []


	def generate_trajectory(self,u):
		y = []
		r = []
		for i in range(len(u)):
			v = np.random.multivariate_normal(np.zeros(self.A.shape[0]),self.covV)
			state = self.A.dot(self.state) + self.B.dot(u[i]) +v
			

			if hasattr(u[i],'__len__'):
				c = self.state.T.dot(self.Q.dot(self.state)) + u[i,:].dot(R.dot(u[i,:]))
			else:
				c = self.state.T.dot(self.Q.dot(self.state)) + self.R*(u[i]**2)
			r.append(c)
			self.state=state
			if hasattr(self.covW, "__len__"):
				w = np.random.multivariate_normal(np.zeros(self.covW.shape[0]),self.covW)

			else:
				w = np.random.normal(0,self.covW)

			y.append(self.C.dot(self.state)+w)

		y = np.array(y)
		r = np.array(r)
		self.input = u
		self.output = y
		self.cost = r
		return u,y,r




def LQG_Smatrix(A,B,Q,R, tol = .001):
	
	eig = np.linalg.eigvals(A)
	assert np.max(np.abs(eig)) <1, 'A matrix is not stable'
	Sold = np.eye(len(A))
	
	invmat = B.T.dot(Sold.dot(B))+R
	if hasattr(invmat,'__len__'):
		invmat = np.linalg.inv(B.T.dot(Sold.dot(B))+R).dot(B.T.dot(Sold))
		invmat = Sold-Sold.dot(B.dot(invmat))
	else:
		invmat = 1/(invmat)
		invmat = Sold-invmat*(Sold.dot(B.dot(B.T.dot(Sold))))
	
	Snew = A.T.dot(invmat.dot(A))+Q
	while np.linalg.norm(Sold-Snew) > tol:
		Sold = np.copy(Snew)
		invmat = B.T.dot(Sold.dot(B))+R
		if hasattr(invmat,'__len__'):
			invmat = np.linalg.inv(B.T.dot(Sold.dot(B))+R).dot(B.T.dot(Sold))
			invmat = Sold-Sold.dot(B.dot(invmat))
		else:
			invmat = 1/(invmat)
			invmat = Sold-invmat*(Sold.dot(B.dot(B.T.dot(Sold))))
		Snew = A.T.dot(invmat.dot(A))+Q

	return Snew




def LQG_Pmatrix(A,C,covW,covV,tol = .001):
	eig = np.linalg.eigvals(A)
	assert np.max(np.abs(eig)) <1, 'A matrix is not stable'

	Pold = np.eye(len(A))

	invmat = C.dot(Pold.dot(C.T))+covW
	if hasattr(invmat,'__len__'):
		invmat = np.linalg.inv(invmat).dot(C.dot(Pold))
		invmat = Pold-Pold.dot(C.T.dot(invmat))
	else:
		invmat = 1/(invmat)
		invmat = Pold-invmat*(Pold.dot(C.T.dot(C.dot(Pold))))
	Pnew = A.T.dot(invmat.dot(A))+covV

	while np.linalg.norm(Pold-Pnew) > tol:
		Pold = np.copy(Pnew)
		invmat = C.dot(Pold.dot(C.T))+covW
		if hasattr(invmat,'__len__'):
			invmat = np.linalg.inv(invmat).dot(C.dot(Pold))
			invmat = Pold-Pold.dot(C.T.dot(invmat))
		else:
			invmat = 1/(invmat)
			invmat = Pold-invmat*(Pold.dot(C.T.dot(C.dot(Pold))))
		Pnew = A.T.dot(invmat.dot(A))+covV

	return Pnew

def LQG_Lmatrix(P,C,covW):

	mat = C.dot(P.dot(C.T))
	if not hasattr(mat,'__len__'):
		mat = 1/(C.dot(P.dot(C.T))+covW)
		return mat*P.dot(C.T)
	else:
		mat = np.linalg.inv(C.dot(P.dot(C.T))+covW)
		return P.dot(C.T.dot(mat))

def LQG_Kmatrix(B,S,R,A):

	mat = B.T.dot(S.dot(B))+R
	if hasattr(mat,'__len__'):
		mat = np.linalg.inv(mat).dot(B.T.dot(S.dot(A)))
		return mat
	else:
		mat = 1/mat
		mat = mat*(B.T.dot(S.dot(A)))
		return mat




if __name__ =='__main__':
	A = np.random.rand(2,2)
	A = A.dot(A.T)
	A /= 10
	B = np.random.rand(2)
	C = np.random.rand(2).reshape((1,2))
	Q = np.random.rand(2,2)
	Q = Q.dot(Q.T)
	R = 1
	covW=1
	covV = np.random.rand(2,2)
	covV = covV.T.dot(covV)
	S = LQG_Smatrix(A,B,Q,R, tol = .001)
	P = LQG_Pmatrix(A,C,covW,covV)
	L =LQG_Lmatrix(P,C,covW)
	K = LQG_Kmatrix(B,S,R,A)

	print(S)
	print(P)

	print(L)
	print(K)

	sys = LQGsystem(A,B,C,covW,covV,Q,R)
	u = np.random.normal(0,1,100)

	sys.generate_trajectory(u)

	print(sys.cost)



















