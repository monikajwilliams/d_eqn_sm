#!/usr/bin/env python -u
import numpy as np

# => Chebshev Operations <= #

# extremal nodes
def cheb_nodes(
    N,
    ):

    return np.cos(np.pi*np.arange(0.0,N+1.0)/N)

def cheb_c(
    N,
    ):

    c = np.ones(N+1)
    c[0] = 2.0
    c[-1] = 2.0

    return c

def cheb_poly(
    x,
    N,
    ):

    T = np.zeros((N+1,len(x)))
    T[0,:] = 1.0
    if N > 0:
        T[1,:] = x
    for n in range(1,N):
        T[n+1,:] = 2.0*x*T[n,:] - T[n-1,:]

    return T


def cheb_trans(
    u,
    N,
    ):

    x = cheb_nodes(N)
    u_k = u(x)
    T = cheb_poly(x,N)
    c = cheb_c(N)
    
    cu_k = u_k/c
    a = np.einsum('k,nk->n',cu_k,T)
    a *= 2.0/N
    a /= c

    return a

def cheb_trans2(
    u_k,
    N,
    ):

    x = cheb_nodes(N)
    T = cheb_poly(x,N)
    c = cheb_c(N)
    
    cu_k = u_k/c
    a = np.einsum('k,nk->n',cu_k,T)
    a *= 2.0/N
    a /= c

    return a

def cheb_colloc(
    x,
    a,
    ):

    #old algorithm
    #N = len(a)-1
    #T = cheb_poly(x,N)
    #u = np.einsum('k,kj->j',a,T)
    #return u

    # Clenshaw algorithm
    b0 = np.zeros_like(x)
    b1 = np.zeros_like(x)
    b2 = np.zeros_like(x)
    N = len(a) - 1
    for i in range(N,0,-1):
        b0 = a[i] + 2.0*x*b1 - b2
        b2 = b1
        b1 = b0
    b0 = 2.0*a[0] + 2.0*x*b1 - b2
    u = 0.5*(b0-b2)
    u[x > 1.0] = 0.0
    u[x < -1.0] = 0.0
    return u

def cheb_D(
    N,
    ):

    c = cheb_c(N)
    x = cheb_nodes(N)

    D = np.zeros((N+1,N+1))    
    for j in range(0,N+1):
        for k in range(0,N+1):
            if j != k:
                D[j,k] = c[j]*(-1)**(j+k)/(c[k]*(x[j]-x[k]))
            else:
                if j == 0:
                    D[j,k] = (2.0*N**2+1.0)/6.0
                elif j == N:
                    D[j,k] = -(2.0*N**2+1.0)/6.0
                else:
                    D[j,k] = -x[j]/(2.0*(1.0-x[j]**2))
                
            
    return D

def cheb_weights(
    N,
    ):

    # Chebyshev-Lobatto Quadrature
    x = cheb_nodes(N)
    w = np.zeros_like(x)
    T = cheb_poly(x,N+1)
    for n in range(1,N+1,2):
        w += 1.0 / n * T[n-1,:]
        w -= 1.0 / (n * (2.0 if n+1 == 0 or n+1 == N else 1.0)) * T[n+1,:]
    w *= 2.0 / N
    w[0] *= 0.5
    w[-1] *= 0.5
    return w

