import numpy as np
from scipy.optimize import minimize
import random
import sys
n=int(sys.argv[1]) # the number of nodes 
W_p=np.random.rand(n,n)
W_n=np.random.rand(n,n)
for i in range(n):
    for j in range(i+1):
        W_p[i,j]=0
        W_n[i,j]=0



def obj (X):
    obj_val=0
    n=W_p.shape[0]
    V=np.reshape(X,(n,n))
    for i in range(n):
        for j in range(i,n):
            obj_val=obj_val+ (W_n[i,j]*( 1-np.dot(V[i,:],V[j,:].T) ) ) + (W_p[i,j]*(np.dot(V[i,:],V[j,:].T)))
    return -obj_val 
def eq_constr(X):
    V=np.reshape(X,(n,n))
    return (np.diag(np.dot(V,V.T))-np.ones(n))
def ineq_constr(X):
    V=np.reshape(X,(n,n))
    return np.dot(V,V.T).flatten()#reshape(n**2,1)


con1={'type': 'eq','fun':eq_constr}
con2={'type': 'ineq','fun':ineq_constr}
x0=np.eye(n).reshape(n**2,1)
res=minimize(obj,x0,method='SLSQP',constraints=[con1,con2])
OPT_X=res.x.reshape(n,n)


# Allocate the corelation clustering 
R1=[]
R2=[]
R3=[]
R4=[]
for i in range (n):
    r1=np.random.normal(0,1,n)
    r2=np.random.normal(0,1,n)
    if np.dot(OPT_X[i,:],r1)<0 and np.dot(OPT_X[i,:],r2)<0:
        R1.append(i)
    elif np.dot(OPT_X[i,:],r1)>0 and np.dot(OPT_X[i,:],r2)<0:
        R2.append(i)
    elif np.dot(OPT_X[i,:],r1)<0 and np.dot(OPT_X[i,:],r2)>0:
        R3.append(i)    
    else:
        R4.append(i)
        
# Calculate the cut value
val=0
for i in range (n):
    for j in range(i+1,n):
        if (i in R1 and j in R1) or (i in R2 and j in R2) or (i in R3 and j in R3) or (i in R4 and j in R4):
            val+=W_p[i,j]
        else:
            val+=W_n[i,j]
            
print("The objective values is:",val)
print ("The 4 clusters are :")
print("R1 :",R1)
print("R2 :",R2)
print("R3 :",R3)
print("R4 :",R4)    