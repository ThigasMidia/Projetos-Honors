from matplotlib import pyplot as plt
import numpy as np

def f(x):
    return np.atan(x)

def df(x):
    return 1/(x**2 + 1)

eixo_x = np.linspace(1e-15,1e-1,100)
x0 = 1
aprox = (f(x0+eixo_x) - f(x0)) / eixo_x
eixo_y = np.abs(df(x0) - aprox)

print(df(x0))
print(aprox)
print(eixo_y)

plt.loglog(eixo_x, eixo_y)
plt.xlim(max(eixo_x), min(eixo_x))
plt.show();


