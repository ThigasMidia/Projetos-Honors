def trapezio(f,a,b,n):
    h = (int(a) + int(b))/int(n)
    valor = 0
    for i in range(1, int(n)):
        valor += f(float(a)+(float(i)*float(h)))
    valor = (((float(valor)*2)+f(float(a))+f(float(b)))*float(h))/2
    return valor

def simpson(f,a,b,n):
    h = (int(a) + int(b))/int(n)
    soma = 0
    x = int(a)
    for i in range(int(n)-1):
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
