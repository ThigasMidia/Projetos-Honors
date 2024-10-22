from matplotlib import pyplot as plt
import numpy as np

def f(x):
    return np.arctan(x)

def df(x):
    return 1/(x**2 + 1)

def ef(x, y):
    return np.cos(x**2)

def fd_error(f,df,x0,h0,hn,n):
    h = np.linspace(h0,hn,n+1)
    aprox = (f(x0+h) - f(x0))/ h
    e = abs(df(x0) - aprox)
    aproxima = (f(x0+0.1) - f(x0)) * 10

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

ode_solver(ef, 0.5, 2, 5, 4, True)
