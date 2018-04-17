import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Return differentiation matrix, central differences
def diffX(M):
    dx = 1/(M+1)
    Dx = -1 * np.eye(M+2, k = -1) + np.eye(M+2, k = 1)
    Dx[0, :3] = [-3, 4, -1]
    Dx[-1, -3:] = [1, -4, 3]
    Dx /= 2*dx
    return Dx

def support_matrices(M):
    # Construction Matrices
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

def assemble_A(u, M, g, Dx, Ξ, Ω, Γ):
    dx = 1/(M+1)
    G = g(Dx.dot(u)**2)
    ξ = Ξ.dot(u)
    ω = Ω.dot(u)
    γ = Γ.dot(u)
    
    diags = (ξ[1:], ω, γ[:-1])
    A = spsp.diags(diags, (-1, 0, 1))/(2*dx**2)
    return A


def echo_output(M, G, Dx, u):
    plt.figure()#figsize = (12,12))
#     plt.subplot(211)
    plt.plot(u)
    
#     plt.subplot(132)
#     plt.plot(Dx.dot(u))
    
#     plt.subplot(212)
#     plt.plot(G)
    
    plt.show()
    
def solve_FE(u0, g, M, T, dt, echo = False):
    dx = 1/(M+1)
    
    r = 1/2 * (dt/dx**2)
    
    U = np.zeros((T, M+2))
    U[0] = u0
    
    Dx = diffX(M)
    
    Ξ, Ω, Γ = support_matrices(M)
    
    for it in range(T-1):
        A = assemble_A(U[it], M, g, Dx, Ξ, Ω, Γ)
        G = g(Dx.dot(U[it])**2)
        
        U[it+1] = U[it] + r * A.dot(U[it])
        if echo:
            if it % (T//30) == 0:
                echo_output(M, G, Dx, U[it])
    return U
    
    
def solve_BE(u0, g, M, T, dt, echo = False):
    dx = 1/(M+1)
    
    r = 1/2 * (dt/dx**2)
    
    U = np.zeros((T, M+2))
    U[0] = u0
    
    Dx = diffX(M)
    
    Ξ, Ω, Γ = support_matrices(M)
    
    for it in range(T-1):
        A = assemble_A(U[it], M, g, Dx, Ξ, Ω, Γ)
        G = g(Dx.dot(U[it])**2)
        
        U[it+1] = spla.spsolve(spsp.identity(M+2) - r * A, U[it])
        
        if echo:
            if it % (T//30) == 0:
                echo_output(M, G, Dx, U[it])
    return U
        
    
def solve_CN(u0, g, M, T, dt, echo = False):
    dx = 1/(M+1)
    
    r = 1/2 * (dt/dx**2)
    
    U = np.zeros((T, M+2))
    U[0] = u0
    
    Dx = diffX(M)
    
    Ξ, Ω, Γ = support_matrices(M)
    
    for it in range(T-1):
        A = assemble_A(U[it], M, g, Dx, Ξ, Ω, Γ)
        G = g(Dx.dot(U[it])**2)
        
        U[it+1] = spla.spsolve(spsp.identity(M+2) + r * A, (spsp.identity(M+2) - r*A).dot(U[it]))
        
        if echo:
            if it % (T//30) == 0:
                echo_output(M, G, Dx, U[it])
    return U
