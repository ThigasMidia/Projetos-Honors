from matplotlib import pyplot as plt
import numpy as np

def fin_diff(f,x,degree,h):

    #iden = "vetor identidade" que sempre muda o 'h' de posicao, sendo h em iden[0] na
    #       primeira iteração, h em iden[1] na segunda e por assim vai
    n = np.size(x);
    iden = np.zeros(n);
    iden[0] = h;

    #dmx = delta-minus x (x - hEj)
    #dpx = delta-plus x (x + hEj)
    if (degree == 1):
        grad = np.zeros(n)
        dmx = x - iden
        dpx = x + iden
        grad[0] = (f(dpx) - f(dmx))/(2*h)
        for i in range(1, n):
            dpx = dpx - iden
            dmx = dmx + iden
            iden[i-1] = 0
            iden[i] = h
            dpx = dpx + iden
            dmx = dmx - iden
            grad[i] = (f(dpx) - f(dmx))/(2*h)

    else:
        grad = np.zeros((n,n))
        for i in range(0,n):
            dmx = x - iden
            dpx = x + iden

            #id2 = identidade iterada no caso de derivada mista (enquanto iden é de hEj, id2 é de hEi) 
            id2 = np.copy(iden)
            for j in range(i, n):
                if(i == j):
                    grad[i,j] = (f(dpx) - 2*f(x) + f(dmx))/h**2
                else:
                    id2[j-1] = 0
                    id2[j] = h
                    grad[i,j] = (f(dpx+id2)-f(dpx-id2)-f(dmx+id2)+f(dmx-id2))/(4*h**2)
                    grad[j,i] = grad[i,j]
            dmx = dmx + iden
            dpx = dpx - iden
            iden[i] = 0
            if(i+1 < n):
                iden[i+1] = h

    return grad


def linesearch(f,x,g,d):
    alpha = 1
    while(f(x+alpha*d(x)) > f(x) + 0.001*alpha*np.dot(g(x),d(x))):
        alpha = alpha*0.5

    return alpha

def f(x):
    return x[0]**4-2*x[0]**2+x[0]-x[0]*x[1]+x[1]**2

def grad(x):
    return np.array([4*x[0]**3-4*x[0]+1-x[1],-x[0]+2*x[1]])

def negGrad(x):
    return np.array([-4*x[0]**3+4*x[0]-1+x[1],x[0]-2*x[1]])

def gd(f,x0,grad,eps = 1e-5,alpha = 0.1,itmax = 10000,fd = False,h = 1e-7,plot = False,search = False):
    x = x0
    k = 0
    while (np.linalg.norm(grad(x)) > eps) and (k < itmax):
        k += 1
        if(search):
            alpha = linesearch(f,x,grad,negGrad)
        x = x - (alpha * grad(x))

    if(plot):
        img = [np.linspace(-1.4,0.5,1000), np.linspace(-0.6,0.5,1000)]
        X, Y = np.meshgrid(img[0],img[1])
        F =  f([X,Y])
        plt.contour(X,Y,F)   
        plt.show()

    return x, k

x,k = gd(f,np.array([0,0]),grad,plot=True)
print(x)
print(k)
