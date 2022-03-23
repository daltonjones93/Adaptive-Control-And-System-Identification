
import numpy as np
import matplotlib.pyplot as plt


#reinforcement learning test

import leastSquares
import hankelEstimator
import LQGsystem


class LQGreinforcementLearning(LQGsystem.LQGsystem):

    def __init__(self,A,B,C,covW,covV,Q,R):
    	LQGsystem.LQGsystem.__init__(self,A,B,C,covW,covV,Q,R)



    def implement_adaptive_control(self,steps_to_update, epochs, hankel_size, num_trajectories, initial_cov):

    	#just design this for siso system then generalize
    	#this also assumes we know the state space dimension
    	self.u = []
    	self.y = []
    	self.r = []
    	self.true_state_trajectory = []
    	
    	n = self.A.shape[0]
    	for i in range(epochs):
    		if i == 0:
    			u_0 = np.random.normal(0,1,steps_to_update)
    			state = np.random.normal(0,1,n)
    			y_0 = []
    			r_0 = []

    			for j in range(steps_to_update):
    				v = np.random.multivariate_normal(np.zeros(self.A.shape[0]),self.covV)
    				r_0.append(state.T.dot(self.Q.dot(state))+(u_0[j]**2)*self.R)
    				state = self.A.dot(state) + self.B*u_0[j]+v
    				w = np.random.normal(0,self.covW)
    				y_0.append(self.C.dot(state)+w)

    			y_0 = np.array(y_0).reshape((steps_to_update,))
    			est = hankelEstimator.HankelEstimator(u,np.array(y_0),self.A.shape[0],hankel_size)
    			est.estimateSystemParameters()

    			#need to estimate some trajectories

    			traj = np.zeros((steps_to_update*num_trajectories,self.A.shape[0]))
    			r_Q_R = np.zeros(steps_to_update*num_trajectories)
    			for ii in range(num_trajectories):
    				for jj in range(steps_to_update):
    					if jj ==0:
    						traj[ii*steps_to_update+jj,:] = np.random.normal(0,initial_cov,self.A.shape[0])
    						r_Q_R[ii*steps_to_update+jj] = r_0[jj]
    					else:
    						state = traj[ii*steps_to_update+jj-1,:]
    						state = est.Ahat.dot(state)+est.Bhat*u_0[jj]+\
    							np.random.multivariate_normal(np.zeros(len(self.covV)),self.covV)
    						r_Q_R[ii*steps_to_update+jj] = r_0[jj]
    						traj[ii*steps_to_update+jj,:] = state

    			Qhat,Rhat = leastSquares.quadratic_estimator_Q_R(traj,u_0,r_Q_R)
    			Sold = np.eye(2)
    			if len(Rhat) ==1:
    				Rhat = Rhat[0,0]

    			Shat = LQGsystem.LQG_Smatrix(est.Ahat,est.Bhat,Qhat,Rhat)
    			Phat = LQGsystem.LQG_Pmatrix(est.Ahat,est.Chat,self.covW,self.covV)
    			Lhat = LQGsystem.LQG_Lmatrix(Phat,est.Chat,self.covW)
    			Khat = LQGsystem.LQG_Kmatrix(est.Bhat,Shat,Rhat,est.Ahat)

    			self.u += u_0.tolist()
    			self.y += y_0.tolist()
    			self.r += list(r_0)
    			

    		else:
    			
    			y_0 = []
    			r_0 = []
    			u_0 = []
    			if i ==1:
    				estState = np.random.normal(0,1,len(self.A))
    				
    			
    			for j in range(steps_to_update):
    				v = np.random.multivariate_normal(np.zeros(self.A.shape[0]),self.covV)
    				ucntrl = -Khat.dot(estState)
    				u_0.append(ucntrl)
    				r_0.append(state.T.dot(self.Q.dot(state))+(ucntrl**2)*self.R)


    				

    				state = self.A.dot(state) + self.B*ucntrl+v 
    				w = np.random.normal(0,self.covW)
    				y_0.append(self.C.dot(state)+w)

    				estState = est.Ahat.dot(state)+est.Bhat*ucntrl + Lhat*(y_0[-1]-\
    					est.Chat.dot(est.Ahat.dot(state)+est.Bhat*ucntrl))


    				

    			y_0 = np.array(y_0).reshape((steps_to_update,))
    			est = hankelEstimator.HankelEstimator(u,np.array(y_0),self.A.shape[0],hankel_size)
    			est.estimateSystemParameters()

    			#need to estimate some trajectories

    			traj = np.zeros((steps_to_update*num_trajectories,self.A.shape[0]))
    			r_Q_R = np.zeros(steps_to_update*num_trajectories)
    			for ii in range(num_trajectories):
    				for jj in range(steps_to_update*(i+1)):
    					if jj ==0:
    						traj[ii*steps_to_update+jj,:] = np.random.normal(0,initial_cov,self.A.shape[0])
    						r_Q_R[ii*steps_to_update+jj] = r_0[jj]
    					else:
    						state = traj[ii*steps_to_update+jj-1,:]
    						state = est.Ahat.dot(state)+est.Bhat*u_0[jj]+\
    							np.random.multivariate_normal(np.zeros(len(self.covV)),self.covV)
    						r_Q_R[ii*steps_to_update+jj] = r_0[jj]
    						traj[ii*steps_to_update+jj,:] = state

    			Qhat,Rhat = leastSquares.quadratic_estimator_Q_R(traj,u_0,r_Q_R)

    			Shat = LQGsystem.LQG_Smatrix(A,B,Q,R)
    			Phat = LQGsystem.LQG_Pmatrix(A,C,covW,covV)
    			Lhat = LQGsystem.LQG_Lmatrix(P,C,covW)
    			Khat = LQGsystem.LQG_Kmatrix(B,S,R,A)

    			self.u += u_0.tolist()
    			self.y += y_0.tolist()
    			self.r += list(r_0)


    			






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

controlSys = LQGreinforcementLearning(A,B,C,covW,covV,Q,R)

u = np.random.normal(0,1,100)

u,y,r = controlSys.generate_trajectory(u)
initial_cov = .1
hankel_size = 20
num_trajectories = 1
controlSys.implement_adaptive_control(1000,10,hankel_size,num_trajectories,initial_cov)


























