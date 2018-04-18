import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spop


def f_init(x, alpha):
    first = 5-np.tanh(alpha*(x-1))-np.tanh(alpha*(x-2))
    second = np.tanh(alpha*(x-4)) + np.tanh(alpha*(x-5)) + 0.1*(np.sin(5*x))**2*np.sin(50*x)
    return first+second


def g(s):
    return 1/(1+s)


def c(U, i, g, dx):
    return g(((U[i+1] - U[i-1])/(2*dx))**2)


def cP(U, i , g, dx):
    return g(((U[i+2] - U[i])/(2*dx))**2)


def cM(U, i, g, dx):
    return g(((U[i] - U[i-2])/(2*dx))**2)


def F(U, i, g, dx, boundary):
    if boundary == 0:
        res = (c(U, i, g, dx) + cP(U, i, g, dx))*(U[i+1] - U[i]) - (cM(U, i, g, dx) + c(U, i, g, dx))*(U[i] - U[i-1])
    elif boundary == -1:
        res = (c(U, i, g, dx) + cP(U, i, g, dx)) * (U[i + 1] - U[i]) - (c(U, i, g, dx) + c(U, i, g, dx)) * (U[i] - U[i - 1])
    else:
        res = (c(U, i, g, dx) + c(U, i, g, dx)) * (U[i + 1] - U[i]) - (c(U, i, g, dx) + c(U, i, g, dx)) * (U[i] - U[i - 1])
    res *= 1/(2*dx**2)
    return res


def f(U, Uprev, dt, g, dx):
    out = [U[0] - Uprev[0]]
    for i in range(1, len(U)-1):
        if i == 1:
            boundary = -1
        elif i == len(U)-2:
            boundary = 1
        else:
            boundary = 0
        out.append(U[i] - dt*F(U, i, g, dx, boundary) - Uprev[i])
    out.append(U[-1] - Uprev[-1])
    return out


M = 100
dx = 6/(M+1)
dt = 0.005
N = 100
alpha = 30

U = np.zeros((N,M+2))
for i in range(M+2):
    U[0][i] = f_init(i*dx, alpha)

for n in range(1, N):
    U[n] = spop.fsolve(f, U[n-1], (U[n-1], dt, g, dx))

print("fig")

plt.figure(0)
plt.plot(U[0])
plt.plot(U[-1])
plt.show()


