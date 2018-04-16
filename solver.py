import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode


def f(x, alpha):
    first = 5-np.tanh(alpha*(x-1))-np.tanh(alpha*(x-2))
    second = np.tanh(alpha*(x-4)) + np.tanh(alpha*(x-5)) + 0.1*(np.sin(5*x))**2*np.sin(50*x)
    return first+second

def g(s):
    return 1/(1+s**2)

alpha = 30
x = np.linspace(0,6,1000)
M = 1000
N = 10000
h = 6/(M+1)
k = 0.01

y0 = [f(h*i, alpha) for i in range(M+2)]
x = [h*i for i in range(M+2)]
t0 = 0

def f(t, y, h):
    dydt = [0]
    for j in range(1, M+1):
        if j == 1:
            c0 = g(np.abs((y[j + 1] - y[j - 1]) / (2 * h)))
            cp = g(np.abs((y[j + 2] - y[j]) / (2 * h)))
            cm = g(np.abs((y[j+1] - y[j - 1]) / (2 * h)))
        elif j == M:
            c0 = g(np.abs((y[j + 1] - y[j - 1]) / (2 * h)))
            cp = g(np.abs((y[j + 1] - y[j-1]) / (2 * h)))
            cm = g(np.abs((y[j] - y[j - 2]) / (2 * h)))
        else:
            c0 = g(np.abs((y[j + 1] - y[j - 1]) / (2 * h)))
            cp = g(np.abs((y[j + 2] - y[j]) / (2 * h)))
            cm = g(np.abs((y[j] - y[j - 2]) / (2 * h)))

        betam = c0 + cm
        betap = c0 + cp
        alpha = -1*(cm + 2*c0 + cp)

        dydt.append(betam*y[j-1] + alpha*y[j] + betap*y[j+1])
    dydt.append(0)
    return dydt

r = ode(f).set_integrator('vode', method='bdf')
r.set_initial_value(y0, t0).set_f_params(h)
t1 = 100
dt = 0.02
while r.successful() and r.t < t1:
    y = r.integrate(r.t+dt)

print('fig')

plt.figure(0)
plt.plot(x, y0)
plt.plot(x, y)
plt.show()

'''
U = np.zeros((N,6*(M+1)))
for i in range(6*M):
    U[0, i] = f(h*i, alpha)

for iter in range(1, N):
    c0 = g(np.abs((U[iter - 1, j + 1] - U[iter - 1, j - 1]) / (2 * h)))
    cp = g(np.abs((U[iter - 1, j + 2] - U[iter - 1, j]) / (2 * h)))
    cm = g(np.abs((U[iter - 1, j] - U[iter - 1, j - 2]) / (2 * h)))
    def f(t, y):
        return (1 / (2 * h ** 2)) * ((c0 + cp) * (U[iter - 1, j + 1] - U[iter - 1, j]) - (cm + c0) * (U[iter - 1, j] - U[iter - 1, j - 1]))
'''