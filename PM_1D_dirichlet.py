import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from schemes import *
import diffusions as func

def before_after_1D(U):
    plt.figure(figsize = (12, 12))
    plt.subplot(211)
    plt.plot(U[0])
    plt.subplot(212)
    plt.plot(U[-1])
    plt.show()

# Create random 1D function
def generate_random1D(M, scale = 50):
    I = np.zeros(M+2)
    s = (M + 2)//5
    for i in range(5):
        I[i*s:(i+1)*s+1] = np.random.randint(0, 2*scale)
    
    I[-1] = I[-2]
    # Add noise to interior points
    I[1:-1] = I[1:-1] + np.random.normal(0, 1, size = M)
    return I

def echo_output(u):
    plt.figure()
    plt.plot(u)
    plt.show()
   
   
# Forward euler, full discretisation 
def solve_FE(u0, diffusion, M, T, dt, echo = False):
    dx = 1/(M+1)
    
    # u-value discretized grid
    U = np.zeros((T, M+2))
    U[0] = u0
    
    
    # Recurring matrices
    Dx = diffX1D(M)
    Ξ, Ω, Γ = support_matrices(M)
    
    for it in range(T-1):
        A = assemble_A1D(U[it], M, diffusion, Dx, Ξ, Ω, Γ)
        U[it+1] = U[it] + dt * A.dot(U[it])
        if echo:
            try: 
                if it % (T//10) == 0:
                    echo_output(U[it])
            except:
                continue
    return U
    
    
def solve_BE(u0, diffusion, M, T, dt, echo = False):
    dx = 1/(M+1)
    
    U = np.zeros((T, M+2))
    U[0] = u0
    
    Dx = diffX1D(M)
    Ξ, Ω, Γ = support_matrices(M)
    
    for it in range(T-1):
        A = assemble_A1D(U[it], M, diffusion, Dx, Ξ, Ω, Γ)
        U[it+1] = spla.spsolve(spsp.identity(M+2) - dt * A, U[it])
        if echo and T > 10:
            if it % (T//10) == 0:
                echo_output(U[it])
    return U
        
    
def solve_CN(u0, diffusion, M, T, dt, echo = False):
    dx = 1/(M+1)
    
    U = np.zeros((T, M+2))
    U[0] = u0
    
    Dx = diffX1D(M)
    Ξ, Ω, Γ = support_matrices(M)
    
    for it in range(T-1):
        A = assemble_A1D(U[it], M, diffusion, Dx, Ξ, Ω, Γ)
        U[it+1] = spla.spsolve(spsp.identity(M+2) + dt * A, (spsp.identity(M+2) - dt*A).dot(U[it]))
        if echo:
            if it % (T//10) == 0:
                echo_output(U[it])
    return U

#################################################################

# Test function
def f(x, alpha, scale = 0.1):
    first = 5-np.tanh(alpha*(x-1))-np.tanh(alpha*(x-2))
    second = np.tanh(alpha*(x-4)) + np.tanh(alpha*(x-5)) + scale*(np.sin(5*x))**2*np.sin(50*x)
    return first+second

if __name__ == "__main__":
    M = 600
    x = np.linspace(0, 6, M+2)

    T = 1000
    dt = 1e-3

    g = func.rational(1)


    U = solve_BE(f(x, 30), g, M , T, dt)
    before_after_1D(U)

    #g_exp = lambda s: np.exp(-s)
    #U_exp = solve_BE(f(x, 30), g_exp, M, T, dt)
    #before_after_1D(U_exp)
