import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def before_after_1D(U):
    plt.figure(figsize = (12, 12))
    plt.subplot(211)
    plt.plot(U[0])
    plt.subplot(212)
    plt.plot(U[-1])
    plt.show()


# Create random 1D function
def generate_random1D(M):
    I = np.zeros(M+2)
    s = (M + 2)//5
    for i in range(5):
        I[i*s:(i+1)*s+1] = 10*np.random.randint(1, 5)
        
    # Add noise to interior points
    I[1:-1] = I[1:-1] + np.random.normal(0, 2, size = M)
    return I


# Return differentiation matrix, central differences
def diffX(M):
    dx = 1/(M+1)

    Dx = -1 * np.eye(M+2, k = -1) + np.eye(M+2, k = 1)
    Dx[0, :3] = [-3, 4, -1]
    Dx[-1, -3:] = [1, -4, 3]
    Dx /= 2*dx
    return Dx

# Construction Matrices for A-matrix
def support_matrices(M):
    Ξ = np.eye(M+2, k = -1) + np.eye(M+2)
    Ξ[0, :2] = 0
    Ξ[-1, -2:] = 0
    
    Ω = - np.eye(M+2, k = -1) - 2 * np.eye(M+2) - np.eye(M+2, k = 1)
    Ω[0, :3] = 0
    Ω[-1, -3:] = 0
    
    Γ = np.eye(M+2) + np.eye(M+2, k = 1)
    Γ[0, :3] = 0
    Γ[-1, -3:] = 0
    
    return Ξ, Ω, Γ

# Assemble A(u). compact differnce scheme matrix
def assemble_A(u, M, diffusion, Dx, Ξ, Ω, Γ):
    dx = 1/(M+1)
    G = diffusion(Dx.dot(u)**2)
    ξ = Ξ.dot(G)
    ω = Ω.dot(G)
    γ = Γ.dot(G)
    
    diags = (ξ[1:], ω, γ[:-1])
    A = spsp.diags(diags, (-1, 0, 1))/(2*dx**2)
    return A


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
    Dx = diffX(M)
    Ξ, Ω, Γ = support_matrices(M)
    
    for it in range(T-1):
        A = assemble_A(U[it], M, diffusion, Dx, Ξ, Ω, Γ)
        U[it+1] = U[it] + dt * A.dot(U[it])
        if echo:
            if it % (T//30) == 0:
                echo_output(U[it])
    return U
    
    
def solve_BE(u0, diffusion, M, T, dt, echo = False):
    dx = 1/(M+1)
    
    U = np.zeros((T, M+2))
    U[0] = u0
    
    Dx = diffX(M)
    Ξ, Ω, Γ = support_matrices(M)
    
    for it in range(T-1):
        A = assemble_A(U[it], M, diffusion, Dx, Ξ, Ω, Γ)
        U[it+1] = spla.spsolve(spsp.identity(M+2) - dt * A, U[it])
        if echo:
            if it % (T//30) == 0:
                echo_output(U[it])
    return U
        
    
def solve_CN(u0, diffusion, M, T, dt, echo = False):
    dx = 1/(M+1)
    
    U = np.zeros((T, M+2))
    U[0] = u0
    
    Dx = diffX(M)
    Ξ, Ω, Γ = support_matrices(M)
    
    for it in range(T-1):
        A = assemble_A(U[it], M, diffusion, Dx, Ξ, Ω, Γ)
        U[it+1] = spla.spsolve(spsp.identity(M+2) + dt * A, (spsp.identity(M+2) - dt*A).dot(U[it]))
        if echo:
            if it % (T//10) == 0:
                echo_output(U[it])
    return U

#################################################################

# Test function
def f(x, alpha):
    first = 5-np.tanh(alpha*(x-1))-np.tanh(alpha*(x-2))
    second = np.tanh(alpha*(x-4)) + np.tanh(alpha*(x-5)) + 0.1*(np.sin(5*x))**2*np.sin(50*x)
    return first+second

if __name__ == "__main__":
    M = 600
    x = np.linspace(0, 6, M+2)

    T = 100
    dt = 1e-3

    g = lambda s: 1/(1+s)
    g_exp = lambda s: np.exp(-s)

    U = solve_FE(f(x, 30), g, M , T, dt)
    U_exp = solve_FE(f(x, 30), g_exp, M, T, dt)
    before_after_1D(U)
    before_efter_1D(U_exp)
