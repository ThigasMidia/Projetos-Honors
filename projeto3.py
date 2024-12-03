from matplotlib import pyplot as plt
import numpy as np
from functools import partial

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
    while(f(x+alpha*d) > f(x) + np.dot(g,d)*alpha*0.001):
        alpha = alpha*0.5
    return alpha

#---------------------------------------------------------------------------------------------
#
#
#
#SEPARACAO ENTRE AS FUNCOES EXTRAS E AS FUNCOES PRINCIPAIS (GRADIENTE, NEWTON E BFGS)
#
#
#
#---------------------------------------------------------------------------------------------


def f(x):
    sum = 0
    for i in range(len(x)-1):
        sum += 100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
    return sum

def gd(f,x0,grad,eps = 1e-5,alpha = 0.1,itmax = 10000,fd = False,h = 1e-7,plot = False,search = False):
    x = x0
    abci = np.array([x[0]])
    orde = np.array([x[1]])
    k = 0

    if(fd):
        grd = fin_diff(f,x,1,h)
        while (np.linalg.norm(grd) > eps) and (k < itmax):
            k += 1
            if(search):
                alpha = linesearch(f,x,grd,-grd)
            x = x - (alpha * grd)
            abci = np.append(abci,[x[0]],axis=0)
            orde = np.append(orde,[x[1]],axis=0)
            grd = fin_diff(f,x,1,h)

    else:
        grd = grad(x)
        while (np.linalg.norm(grd) > eps) and (k < itmax):
            k += 1
            if(search):
                alpha = linesearch(f,x,grd,-grd)
            x = x - (alpha * grd)
            abci = np.append(abci,[x[0]],axis=0)
            orde = np.append(orde,[x[1]],axis=0)
            grd = grad(x)


    if(plot):
        yMin = np.min(orde)
        yMax = np.max(orde)
        xMin = np.min(abci)
        xMax = np.max(abci)
        img = [np.linspace(xMin-0.2*(xMax-xMin),xMax+0.2*(xMax - xMin),1000), np.linspace(yMin-0.2*(yMax-yMin),yMax+0.2*(yMax-yMin),1000)]
        X, Y = np.meshgrid(img[0],img[1])
        F =  f([X,Y])
        plt.contour(X,Y,F,50)
        plt.plot(abci,orde,'blue') 
        plt.plot(abci,orde,'bo')
        plt.show()

    return x, k

def newton(f,x0,grad,hess,eps = 1e-5,alpha = 0.1,itmax = 10000,fd = False,h = 1e-7,plot = False,search = False):
    x = x0
    k = 0
    abci = np.array([x[0]])
    orde = np.array([x[1]])

    if(fd):
        g = fin_diff(f,x,1,h)
    else:
        g = grad(x)

    while (np.linalg.norm(g) > eps) and(k < itmax):
        k += 1
        if(fd):
            H = fin_diff(f,x,2,h)
        else:
            H = hess(x)
    
        try:
            d = np.linalg.solve(H, -g)

        except:
            H = H*0.9 + np.identity(np.size(x)) * 0.1

        while(np.dot(d,g) > -1e-3*np.linalg.norm(g)*np.linalg.norm(d)):
            H = H*0.9 + np.identity(np.size(x))*0.1
            #RESOLVE
            d = np.linalg.solve(H,-g)
        
        x = x + d
        abci = np.append(abci,[x[0]],axis=0)
        orde = np.append(orde,[x[1]],axis=0)
     
        if(fd):
            g = fin_diff(f,x,1,1e-5)
        else:
            g = grad(x)
 
    if(plot):
        yMin = np.min(orde)
        yMax = np.max(orde)
        xMin = np.min(abci)
        xMax = np.max(abci)
        img = [np.linspace(xMin-0.2*(xMax-xMin),xMax+0.2*(xMax - xMin),1000), np.linspace(yMin-0.2*(yMax-yMin),yMax+0.2*(yMax-yMin),1000)]
        X, Y = np.meshgrid(img[0],img[1])
        F =  f([X,Y])
        plt.contour(X,Y,F,50)
        plt.plot(abci,orde,'blue') 
        plt.plot(abci,orde,'bo')
        plt.show()       

    return x, k
        
def bfgs(f,x0,grad,eps = 1e-5,alpha = 0.1, itmax = 10000, fd = False,h = 1e-7,plot = False,search = False):
    x = x0
    k = 0
    g = grad(x)
    H = np.identity(np.size(x))
    abci = np.array([x[0]])
    orde = np.array([x[1]])
    while (np.linalg.norm(g) > eps) and (k < itmax):
        k += 1
        y = g
        s = x
        d = -np.matmul(H,g)
        while (np.dot(d,g) > np.linalg.norm(g)*np.linalg.norm(d)*-0.001):
            H = 0.9*H + 0.1*np.identity(np.size(x))
            d = -np.matmul(H,g)
        if(search):
            alpha = linesearch(f,x,g,d)
        x = x + alpha*d
        abci = np.append(abci,[x[0]],axis=0)
        orde = np.append(orde,[x[1]],axis=0)
        g = grad(x)
        y = g - y
        s = x - s
        #SALVAGUARDA DE DIVISAO POR 0
        if(np.dot(s,y) == 0):
            H = np.identity(np.size(x))
        else:
            H = H + (np.outer(s,s)*(np.dot(s,y)+np.dot(y,np.matmul(H,y))))/np.dot(s,y)**2 - (np.matmul(H,np.outer(y,s)) + np.matmul(np.outer(s,y),H))/np.dot(s,y)

    if(plot): 
        yMin = np.min(orde)
        yMax = np.max(orde)
        xMin = np.min(abci)
        xMax = np.max(abci)
        img = [np.linspace(xMin-0.2*(xMax-xMin),xMax+0.2*(xMax - xMin),1000), np.linspace(yMin-0.2*(yMax-yMin),yMax+0.2*(yMax-yMin),1000)]
        X, Y = np.meshgrid(img[0],img[1])
        F =  f([X,Y])
        plt.contour(X,Y,F,50)
        plt.plot(abci,orde,'blue') 
        plt.plot(abci,orde,'bo')
        plt.show()     

    return x, k

print(x)
print(k)
