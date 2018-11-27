import numpy as np
from scipy.optimize import minimize
import random
import sys
# Initialize
n=int(sys.argv[1]) # the number of nodes 
W=np.random.rand(n,n)
for i in range(n):
    for j in range(i+1):
        W[i,j]=0
def obj (X):
    obj_val=0
    n=W.shape[0]
    V=np.reshape(X,(n,n))
    for i in range(n):
        for j in range(i,n):
            obj_val=obj_val+ (W[i,j]*( 1-np.dot(V[i,:],V[j,:].T) ) )
    return -obj_val/2 
def constr(X):
    V=np.reshape(X,(n,n))
    return (np.diag(np.dot(V,V.T))-np.ones(n))


con={'type': 'eq','fun':constr}
x0=np.eye(n).reshape(n**2,1)
res=minimize(obj,x0,method='SLSQP',constraints=con)
OPT_X=res.x.reshape(n,n)

OPT_X=res.x.reshape(n,n)
U=[]
V=[]
U1=[]
V1=[]
xx=np.eye(n)
for i in range(n):
    r=np.random.normal(0,1,n)
    if np.dot(OPT_X[i,:],r)<0:
        U.append(i)
    else:
        V.append(i)
    if r[i]<0:
        U1.append(i)
    else:
        V1.append(i)    
print("The cut is :")
print("U:",U)
print("V:",V)
OBJ_VAL=0
for u in U:
    for v in V:
        if u<v :
            OBJ_VAL+=W[u,v]
        else:
            OBJ_VAL+=W[v,u]
print("The cut value:",OBJ_VAL)
