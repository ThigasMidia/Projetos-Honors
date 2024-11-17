from matplotlib import pyplot as plt
import numpy as np

'''
def linesearch(f,x,g,d):
    alpha = 1
    while(f(x+alpha*d) > f(x) + alpha*1e-3*d*g(x)):
        alpha /= 2

    return alpha
'''

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
        '''
        if(search):
            alpha = linesearch(f,x,grad,neg_grad)
        '''
        x = x - (alpha * grad(x))

    if(plot):
        img = [np.linspace(-1.5,0.5,1000), np.linspace(-0.7,0.5,1000)]
        X, Y = np.meshgrid(img[0],img[1])
        F =  f([X,Y])
        plt.contour(X,Y,F)   
        plt.show()

    return x, k

x,k = gd(f,np.array([0,0]),grad,search=True)
print(x)
print(k)
