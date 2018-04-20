import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt



# Return differentiation matrix, central differences
def diffX1D(M, domain = (0, 1)):
    dx = (domain[1] - domain[0])/(M+1)

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
def assemble_A1D(u, M, diffusion, Dx, Ξ, Ω, Γ, domain = (0, 1)):
    dx = (domain[1] - domain[0])/(M+1)
    G = diffusion(Dx.dot(u)**2)
    ξ = Ξ.dot(G)
    ω = Ω.dot(G)
    γ = Γ.dot(G)
    
    diags = (ξ[1:], ω, γ[:-1])
    A = spsp.diags(diags, (-1, 0, 1))/(2*dx**2)
    return A

# Finite difference matrix Y-direction
def diffX(M, N, domain = (0, 1)):
    dx = (domain[1] - domain[0])/(M+1)
    Bx = (-1 * np.eye(M+2, k = -1) + np.eye(M+2, k = 1))
    Bx[0, :3] = [-3, 4, -1]
    Bx[-1, -3:] = [1, -4, 3]
    Bx /= (2*dx)
    return spsp.block_diag([Bx]*(N+2))



# Finite difference matrix Y-direction
def diffY(M, N, domain = (0, 1)):
    K = (M+2)*(N+2)
    dy = (domain[1] - domain[0])/(N+1)
    Dy = spsp.diags((-1, 1), (-M-2, M+2), shape = (K, K), format = "lil")
    Dy[:(M+2), :3*(M+2)] = spsp.hstack((-3*spsp.identity(M+2), 4*spsp.identity(M+2), -spsp.identity(M+2)))
    Dy[-(M+2):, -3*(M+2):] = spsp.hstack((spsp.identity(M+2), -4*spsp.identity(M+2), 3*spsp.identity(M+2)))
    return Dy.tocsr()/(2*dy)



# Matrices to help constuct A(u)
def support_matrices_X(M, N):
    K = (M+2)*(N+2)

    #Block matrix for Ξx
    Xx = np.eye(M+2, k = -1) + np.eye(M+2)
    Xx[0, :2] = 0
    Xx[-1, -2:] = 0

    #Block matrix for Ωx
    Mx = -np.eye(M+2, k = -1) -2*np.eye(M+2) - np.eye(M+2, k = 1)
    Mx[0, :2] = 0
    Mx[-1, -2:] = 0

    #Block matrix for Γx
    Fx = np.eye(M+2) + np.eye(M+2, k = 1)
    Fx[0, :2] = 0
    Fx[-1, -2:] = 0
    
    Ξx = spsp.block_diag([Xx]*(N+2))
    Ωx = spsp.block_diag([Mx]*(N+2))
    Γx = spsp.block_diag([Fx]*(N+2))
    
    return Ξx, Ωx, Γx



# Matrices to help constuct A(u)
def support_matrices_Y(M, N):
    K = (M+2)*(N+2)
    Ξy = spsp.diags((np.ones((M+2)*(N+1)), np.ones(K)), (-M-2, 0), format = "lil")
    Ωy = spsp.diags((-np.ones((M+2)*(N+1)), -2*np.ones(K), -np.ones((M+2)*(N+1))), (-M-2, 0, M+2), format = "lil")
    Γy = spsp.diags((np.ones(K), np.ones((M+2)*(N+1))), (0, M+2), format = "lil")
    Ξy[0::M+2, 0::M+2] = 0
    Ξy[M+1::M+2, M+1::M+2] = 0
    Ξy[0:M+2] = 0
    Ξy[-M-2:] = 0

    Ωy[0::M+2, 0::M+2] = 0
    Ωy[M+1::M+2, M+1::M+2] = 0
    Ωy[0:M+2] = 0
    Ωy[-M-2:] = 0

    Γy[0::M+2, 0::M+2] = 0
    Γy[M+1::M+2, M+1::M+2] = 0
    Γy[0:M+2] = 0
    Γy[-M-2:] = 0
    return Ξy.tocsr(), Ωy.tocsr(), Γy.tocsr()



# Put together A(u), spatial difference scheme matrix.
def assemble_A(u, M, N, g, Dx, Dy, Ξx, Ωx, Γx, Ξy, Ωy, Γy, domain=((0, 1), (0, 1))):
    dx = (domain[0][1] - domain[0][0])/(M+1)
    dy = (domain[1][1] - domain[1][0])/(N+1)
    G = g(Dx.dot(u)**2 + Dy.dot(u)**2)
    
    ξx = Ξx.dot(G)
    ωx = Ωx.dot(G)
    γx = Γx.dot(G)

    ξy = Ξy.dot(G)
    ωy = Ωy.dot(G)
    γy = Γy.dot(G)
    
    x_diags = (ξx[1:], ωx, γx[:-1])
    y_diags = (ξy[(M+2):], ωy, γy[:-(M+2)])
    
    Ax = spsp.diags(x_diags, (-1, 0, 1))
    Ay = spsp.diags(y_diags, (-(M+2), 0, M+2))
    A = 1/2*(Ax/dx**2 + Ay/dy**2)
    return A
