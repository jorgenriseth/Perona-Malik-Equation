import numpy as np
import matplotlib.pyplot as plt 
from PM_1D_dirichlet import diffX


def heat(c = 1):
    return lambda s: c * np.ones_like(s)

def rational(c = 1):
    return lambda s: 1/(1+s/c**2)

def diff_rational(c = 1):
    return lambda s: - s/c**2 * rational(c)(s)**2

def exponential(c = 1):
    return lambda s: np.exp(-s/(2 * c**2))

def tukeys(c = 1):
    S = 2*c**2
    def g(s):
        ind = s <= S
        out = np.zeros_like(s)
        out[ind] = 1/2 * (1 - s[ind] / S)**2
        return out
    return g

def weickert(c = 1):
    def g(s):
        ind = s != 0
        out = np.ones_like(s)
        out[ind] = 1 - np.exp(-3.31488*c**8/s[ind]**4)
        return out 
    return g

def zhichang(c = 1):
    def g(s):
        a = lambda s: 2 - 2/(1+s/c**2)
        return 1/(1 + (np.sqrt(s)/c)**a(s))
    return g

def flux(func, c = 1):
    g = func(c)
    return lambda s: s**(1/2) * g(s)

def choose_function(funcname, c = 1):
    if funcname in [0, "heat"]:
        return heat(c)
    if funcname in [1 , "rational"]:
        return rational(c)
    if funcname in [2, "exponential"]:
        return exponential(c)
    if funcname in [3, "tukeys"]:
        return tukeys(c)
    if funcname in [4, "weickert"]:
        return weickert(c)
    if funcname in [5, "zhichang"]:
        return zhichang(c)
    raise Exception("Invalid diffusion function")

if __name__ == "__main__":
    M = 399
    x = np.linspace(0, 6, M+2)

    plt.figure(figsize=(12, 8))
    c = 1
    plt.subplot(211)
    plt.plot(x, rational(c)(x), label = r"$g_1(\Vert\nabla u\Vert)$")
    plt.plot(x, exponential(c)(x), label = r"$g_2(\Vert\nabla u\Vert)$")
    plt.plot(x, tukeys(c)(x), label = r"$g_3(\Vert\nabla u\Vert)$")
    plt.plot(x, zhichang(c)(x), label = r"$g_4(\Vert\nabla u\Vert)$")
    plt.plot(x, weickert(c)(x), label = r"$g_5(\Vert\nabla u\Vert)$")
    plt.ylim(0, 1.03)
    plt.xlim(0, np.max(x))
    plt.xlabel(r"$\Vert\nabla u\Vert)$")
    plt.ylabel(r"$g_1(\Vert\nabla u\Vert)$")
    plt.legend()


    c = 0.5
    plt.subplot(212)
    plt.plot(x, rational(c)(x), label = r"$g_1(\Vert\nabla u\Vert)$")
    plt.plot(x, exponential(c)(x), label = r"$g_2(\Vert\nabla u\Vert)$")
    plt.plot(x, tukeys(c)(x), label = r"$g_3(\Vert\nabla u\Vert)$")
    plt.plot(x, zhichang(c)(x), label = r"$g_4(\Vert\nabla u\Vert)$")
    plt.plot(x, weickert(c)(x), label = r"$g_5(\Vert\nabla u\Vert)$")
    plt.ylim(0, 1.03)
    plt.xlim(0, np.max(x))
    plt.xlabel(r"$\Vert\nabla u\Vert)$")
    plt.ylabel(r"$g_1(\Vert\nabla u\Vert)$")
    plt.legend()

    plt.savefig("./figures/diff_functions.png")
    plt.show()

    plt.figure(figsize=(12, 8))
    c = 1
    plt.subplot(211)
    plt.plot(x, flux(rational, c)(x), label = r"$g_1(\Vert\nabla u\Vert)$")
    plt.plot(x, flux(exponential, c)(x), label = r"$g_2(\Vert\nabla u\Vert)$")
    plt.plot(x, flux(tukeys, c)(x), label = r"$g_3(\Vert\nabla u\Vert)$")
    plt.plot(x, flux(zhichang, c)(x), label = r"$g_4(\Vert\nabla u\Vert)$")
    plt.plot(x, flux(weickert, c)(x), label = r"$g_5(\Vert\nabla u\Vert)$")
    plt.ylim(0, 1.03)
    plt.xlim(0, np.max(x))
    plt.xlabel(r"$\Vert\nabla u\Vert)$")
    plt.ylabel(r"$g_1(\Vert\nabla u\Vert)$")
    plt.legend()


    c = 0.5
    plt.subplot(212)
    plt.plot(x, flux(rational, c)(x), label = r"$g_1(\Vert\nabla u\Vert)$")
    plt.plot(x, flux(exponential, c)(x), label = r"$g_2(\Vert\nabla u\Vert)$")
    plt.plot(x, flux(tukeys, c)(x), label = r"$g_3(\Vert\nabla u\Vert)$")
    plt.plot(x, flux(zhichang, c)(x), label = r"$g_4(\Vert\nabla u\Vert)$")
    plt.plot(x, flux(weickert, c)(x), label = r"$g_5(\Vert\nabla u\Vert)$")
    plt.ylim(0, 1.03)
    plt.xlim(0, np.max(x))
    plt.xlabel(r"$\Vert\nabla u\Vert)$")
    plt.ylabel(r"$g_1(\Vert\nabla u\Vert)$")
    plt.legend()

    plt.savefig("./figures/flux_functions.png")
    plt.show()

    Dx = diffX(M)
    plt.figure(figsize=(12, 8))
    c = 1
    plt.subplot(211)
    plt.plot(x, Dx.dot(flux(rational, c)(x)), label = r"$g_1(\Vert\nabla u\Vert)$")
    plt.plot(x, Dx.dot(flux(exponential, c)(x)), label = r"$g_2(\Vert\nabla u\Vert)$")
    plt.plot(x, Dx.dot(flux(tukeys, c)(x)), label = r"$g_3(\Vert\nabla u\Vert)$")
    plt.plot(x, Dx.dot(flux(zhichang, c)(x)), label = r"$g_4(\Vert\nabla u\Vert)$")
    plt.plot(x, Dx.dot(flux(weickert, c)(x)), label = r"$g_5(\Vert\nabla u\Vert)$")
    plt.xlim(0, 2)
    plt.xlabel(r"$\Vert\nabla u\Vert)$")
    plt.ylabel(r"$g_1(\Vert\nabla u\Vert)$")
    plt.legend()


    c = 0.5
    plt.subplot(212)
    plt.plot(x, Dx.dot(flux(rational, c)(x)), label = r"$g_1(\Vert\nabla u\Vert)$")
    plt.plot(x, Dx.dot(flux(exponential, c)(x)), label = r"$g_2(\Vert\nabla u\Vert)$")
    plt.plot(x, Dx.dot(flux(tukeys, c)(x)), label = r"$g_3(\Vert\nabla u\Vert)$")
    plt.plot(x, Dx.dot(flux(zhichang, c)(x)), label = r"$g_4(\Vert\nabla u\Vert)$")
    plt.plot(x, Dx.dot(flux(weickert, c)(x)), label = r"$g_5(\Vert\nabla u\Vert)$")
    plt.xlim(0, 2)
    plt.xlabel(r"$\Vert\nabla u\Vert)$")
    plt.ylabel(r"$g_1(\Vert\nabla u\Vert)$")
    plt.legend(loc = 1)

    plt.savefig("./figures/flux_derivatives.png")
    plt.show()
