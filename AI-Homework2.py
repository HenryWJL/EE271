#import matplotlib
import numpy as np
#from matplotlib import pyplot as plt

# Task1 Writing a Python class

class Filter:
    
    def __init__(self,A,B,H,Q,R):
        self.A=A
        self.B=B
        self.H=H
        self.Q=Q
        self.R=R
        
    
    def filter(self,x_1,P_1,z):       
        n=P_1.shape[0]
        
        x_=self.A@x_1+self.B@u_1
        P_=self.A@P_1@(self.A.T)+self.Q
        
        K=P_@(self.H.T)@(np.linalg.inv(self.H@P_@(self.H.T)+self.R))
        x=x_+K@(z-self.H@x_)
        P=(np.identity(n)-K@self.H)@P_
        
    
        return x,P
        
            

# Task2 Testing the filter

# Here you can change constants and arguments
                                               # Size
A=np.array([[1,2,3],[4,5,6],[7,8,9]])          # nxn
B=np.array([[1,2,3,4],[3,4,5,6],[5,6,7,8]])    # nxl
H=np.array([[1,2,3],[4,5,6]])                  # mxn
Q=np.array([[6,-3,1],[-3,2,0],[1,0,4]])        # nxn
R=np.array([[1,2],[2,4]])                      # mxm
P_1=np.ones((3,3))                             # nxn
x_1=np.array([1,2,3])                          # nx1 x^k-1
u_1=np.array([0,0,0,0])                        # lx1 uk-1
x0=np.array([3,4,5])                           # nx1 xk-1



# The following is the test part

'''matplotlib.rc('figure',figsize=(20,20))
matplotlib.rc('font',size=20)
matplotlib.rc('axes',grid=False)
matplotlib.rc('axes',facecolor='white')'''


def iteration(x_1,u_1,P_1,x0,m):   # m represents the loops
    
    test=Filter(A,B,H,Q,R)
    
    if(m<=0):
        print("Invalid")
        return (0,0,0)
    
    elif(m==1):
        w=np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
        v=np.random.multivariate_normal(np.zeros(R.shape[0]), R)
        x1=A@x0+B@u_1+w
        z=H@x1+v
        x,P=test.filter(x_1,P_1,z)
        return x,P,x1
    
    else:
        x,P,x1=iteration(x_1, u_1, P_1, x0, 1)
        m-=1
        return iteration(x, u_1, P, x1, m)


newx,newP,newx1=iteration(x_1, u_1, P_1, x0, 20)  

print(newx)               # x^k

print(newx1)              # xk

print(newx-newx1)         # x^k-xk


    
    
    
    
    
    
    




                         


