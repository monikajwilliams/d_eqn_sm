#!/usr/bin/env python -u
import os,sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import util
import numpy as np
import matplotlib.pyplot as plt

def u0gauss(
    x,
    y,
    z,
    ):

    # Mass is (pi / alpha)^3/2
    return np.exp(-5.0 * ((x)**2 + y**2 + z**2))

def u0annulus(
    x,
    y,
    z,
    Ro,
    Ri,
    ):

    Rm = (Ri + Ro)/2.0 
    Rd = (Ro-Ri)

    r = np.sqrt(x**2 + y**2 + z**2)
    fx = np.exp(-Rd * (r-Rm)**2)

    return fx
def u0r(
    x,
    y,
    z,
    Ro,
    Ri,
    ):

    # Mass is (pi / alpha)^3/2
    return 1.0/(np.sqrt(x**2 + y**2 + z**2))


def test(

    u0=u0annulus, #function to evaluate initial density as fxn of x,y,z
    u1=u0r, #potential of mean force as a fxn of x,y,z
    k=1000.0,#reaction constant for center
    Ri=1.0, # inner radius of sphere
    Ro=6.0, # outer radius of sphere
    Ncheb=32, #order of Chebyshev radial basis
    Lsh=10, #order of spherical harmonic angular basis
    Nleb=302, #number of points in the Lebedev grid
    v=1.0, #diffusion coefficient
    dt=0.1, #timestep
    tmax=40.0, #maximum time
    ):

    R = Ro-Ri
    # lebedev points
    leb = util.Lebedev.build(Nleb)
   
    # chebyshev nodes 
    xcheb = util.cheb_nodes(Ncheb)
    # chebyshev weights [-1,+1]
    wcheb = util.cheb_weights(Ncheb)
    # chebyshev nodes mapped to r
    rcheb = (Ro - Ri)/2.0 * (xcheb + 1.0) + Ri
    # chebyshev radal weights [Ri, Ro]
    vcheb = wcheb * R/2.0 * rcheb**2

    # time slices 
    ts = np.arange(0.0,tmax + 0.5*dt,dt)

    # state vector initialization
    u = {}
    uf = {}
    for l in range(0,Lsh+1):
        for m in range(-l,l+1):
            u[(l,m)] = np.zeros((len(ts),Ncheb+1))
            uf[(l,m)] = np.zeros((len(ts),Ncheb+1))

    # initial conditions
    theta = leb.theta 
    phi = leb.phi 
    wleb = leb.w
    Y = util.sh(theta,phi,Lsh)
    ur = np.zeros(len(rcheb))
    ru = np.zeros(len(rcheb))
    for rind,r in enumerate(rcheb):
        x,y,z = util.sphere_to_cart(r,theta,phi) 
        u0val = u0(x,y,z,Ro,Ri)
        ufval = u1(x,y,z,Ro,Ri)
        #saving radial positions and energies for test plotting
        ru[rind] =  np.sqrt(x**2 + y**2 + z**2)[0]
        ur[rind] = ufval[0]
        for l in range(0,Lsh+1):
            for m in range(-l,l+1):
                Yval = Y[(l,m)]
                u[(l,m)][0,rind] = np.sum(wleb*Yval*u0val)
                uf[(l,m)][0,rind] = np.sum(wleb*Yval*ufval)

    # establish the propagator (Crank Nicolson)
    a = v*dt/2.0
    I = np.eye(Ncheb+1)
    D = util.cheb_D(Ncheb)
    M = 2.0/R*D
    K0 = np.dot(np.diag(1.0/rcheb**2),np.dot(M,np.dot(np.diag(rcheb**2),M)))
 
    # calculate the force
    K1 = D
    # Propagate!
    for l in range(0,Lsh+1):
        for m in range(-l,l+1):
            for ind in range(1,len(ts)):
                #uf[(l,m)][ind,:] = np.einsum('ij,j->i',K1[l],uf[(l,m)][ind-1,:])
                uf[(l,m)][ind,:] = np.dot(K1[l],uf[(l,m)][ind-1,:])

    print np.shape(uf[1])
    print np.shape(ur)
    plt.clf()
    plt.plot(ru,ur)
    plt.plot(ru,uf)
    plt.show()
    exit()


test()
