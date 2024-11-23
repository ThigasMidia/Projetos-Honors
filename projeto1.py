def trapezio(f,a,b,n):
    h = (a + b)/n
    valor = 0
    for i in range(1, n):
        valor += f(a+i*h)
    valor = (((valor*2)+f(a)+f(b))*h)/2
    return valor

def simpson(f,a,b,n):
    h = (a + b)/n
    soma = 0
    x = a
    for i in range(n-1):
        x += h
        valor = f(x)
        if i%2==0:
            soma += 4*valor
        else:
            soma += 2*valor

    soma += f(a)+f(b)
    soma *= h/3
    print(soma)
    return soma
