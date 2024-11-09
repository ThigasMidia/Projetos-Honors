from matplotlib import pyplot as plt
import numpy as np

def fd_error(f,df,x0,h0,hn,n):
    h = np.logspace(np.log10(h0),np.log10(hn),n+1)
    e = abs(df(x0) -  (f(x0+h) - f(x0))/ h)

    plt.loglog(h, e)
    plt.gca().invert_xaxis()
    plt.show()

def ode_solver(f,x0,y0,xn,n,plot):
    x = np.linspace(x0, xn, n+1)
    h = x[2] - x[1]
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(1, n+1):
        y[i] = y[i-1] + h*f(x[i-1], y[i-1])
    
    if(plot):
        plt.plot(x, y)
        plt.show()
    
    return x, y
