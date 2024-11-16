from matplotlib import pyplot as plt
import numpy as np

def f(x):
    return x[0]**4-2*x[0]**2+x[0]-x[0]*x[1]+x[1]**2

def grad(x):
    return np.array([4*x[0]**3-4*x[0]+1-x[1],-x[0]+2*x[1]])

def gd(f,x0,grad,eps = 1e-5,alpha = 0.1,itmax = 10000,fd = False,h = 1e-7,plot = False,search = False):
    x = x0
    k = 0
    a = x - (alpha * grad(x))
    while (np.linalg.norm(grad(x)) > eps) and (k < itmax):
        k += 1
        x = x - (alpha * grad(x))
    
    return x, k

x,k = gd(f,np.array([0,0]),grad)
print(x)
print(k)
