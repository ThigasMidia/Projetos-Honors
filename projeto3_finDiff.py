import numpy as np

def f(x):
    return np.sin(x[0]*x[1])*np.cos(x[1]**2)

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

g = fin_diff(f,np.array([-1,2]),1,1e-5)
print(g)
H = fin_diff(f,np.array([-1,2]),2,1e-5)
print(H)
