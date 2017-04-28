#!/usr/bin/env python -u
from mpl_toolkits.mplot3d import Axes3D
import util
import numpy as np
import matplotlib.pyplot as plt

class Sphere(object):

    def __init__(
        self,
        u0=None, #function to evaluate initial density as fxn of x,y,z
        R=5.0, #radius of sphere
        Ncheb=32, #order of Chebyshev radial basis
        Lsh=10, #order of spherical harmonic angular basis
        Nleb=302, #number of points in the Lebedev grid
        v=1.0, #diffusion coefficient
        dt=0.1, #timestep
        tmax=4.0, #maximum time
        ):

        self.R = R
        self.Ncheb = Ncheb
        self.Lsh = Lsh
        self.Nleb = Nleb
        self.v = v
        self.dt = dt
        self.tmax = tmax
        
        # lebedev points
        self.leb = util.Lebedev.build(self.Nleb)
       
        # chebyshev nodes 
        self.xcheb = util.cheb_nodes(self.Ncheb)
        # chebyshev weights [-1,+1]
        self.wcheb = util.cheb_weights(self.Ncheb)
        # chebyshev nodes mapped to r [0, R]
        self.rcheb = self.R/2.0 * (self.xcheb + 1.0)
        # chebyshev radal weights [0, R]
        self.vcheb = self.wcheb * self.R/2.0 * self.rcheb**2
   
        # time slices 
        self.ts = np.arange(0.0,self.tmax + 0.5*self.dt,self.dt)

        # state vector initialization
        self.u = {}
        for l in range(0,self.Lsh+1):
            for m in range(-l,l+1):
                self.u[(l,m)] = np.zeros((len(self.ts),self.Ncheb+1))

        # initial conditions
        theta = self.leb.theta 
        phi = self.leb.phi 
        wleb = self.leb.w
        Y = util.sh(theta,phi,self.Lsh)
        for rind,r in enumerate(self.rcheb):
            x,y,z = util.sphere_to_cart(r,theta,phi) 
            u0val = u0(x,y,z)
            for l in range(0,self.Lsh+1):
                for m in range(-l,l+1):
                    Yval = Y[(l,m)]
                    self.u[(l,m)][0,rind] = np.sum(wleb*Yval*u0val)

        # establish the propagator (Crank Nicolson)
        self.a = self.v*self.dt/2.0
        I = np.eye(self.Ncheb+1)
        D = util.cheb_D(self.Ncheb)
        M = 2.0/self.R*D
        K0 = np.dot(np.diag(1.0/self.rcheb**2),np.dot(M,np.dot(np.diag(self.rcheb**2),M)))
        K = {}
        for l in range(0,self.Lsh+1):
            K[l] = K0 - np.diag(l*(l+1.0)/self.rcheb**2)
        self.U = {}
        for l in range(0,self.Lsh+1):
            L = I - self.a*K[l]
            R = I + self.a*K[l]
            # Boundary Conditions
            R[0,:] = 0.0
            R[-1,:] = 0.0
            L[0,:] = M[0,:]
            if l == 0:
                L[-1,:] = M[-1,:]
            else:
                L[-1,:] = 0.0
                L[-1,-1] = 1.0
            # Propagator 
            self.U[l] = np.linalg.solve(L,R)

        # Propagate!
        for l in range(0,self.Lsh+1):
            for m in range(-l,l+1):
                for ind in range(1,len(self.ts)):
                    self.u[(l,m)][ind,:] = np.einsum('ij,j->i',self.U[l],self.u[(l,m)][ind-1,:])
        
             
        
    @property
    def xyzw(
        self,
        ):
    
        val = np.zeros(((self.Ncheb+1)*self.Nleb,4))     
        for rind,r in enumerate(self.rcheb):
            val[rind*self.Nleb:(rind+1)*self.Nleb,:3] = r*self.leb.xyz
            val[rind*self.Nleb:(rind+1)*self.Nleb,3] = self.leb.w * self.vcheb[rind]

        return val

    def rad_colloc(
        self,
        tind,
        r,
        ):

        x = 2.0 * r / self.R - 1.0
        vals = {}  
        for l in range(0,self.Lsh+1):
            for m in range(-l,l+1):
                avals = util.cheb_trans2(self.u[(l,m)][tind,:],self.Ncheb)
                vals[(l,m)] = util.cheb_colloc(x,avals)

        return vals 

    def colloc(
        self,
        tind,
        x,
        y,  
        z,
        ):

        r,theta,phi = util.cart_to_sphere(x,y,z)
        Y = util.sh(theta,phi,self.Lsh)
        R = self.rad_colloc(tind,r)

        val = np.zeros_like(x)  
        for l in range(0,self.Lsh+1):
            for m in range(-l,l+1):
                val += Y[(l,m)] * R[(l,m)]
            
        return val

# => Testing scope <= #       

def plot_grid(
    sphere,
    ):

    xyzw = sphere.xyzw
    xs = xyzw[:,0]
    ys = xyzw[:,1]
    zs = xyzw[:,2]
    ws = xyzw[:,3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=ws)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plot_rads(
    sphere,
    tind,
    filename,
    ):

    rs = np.linspace(0.0,sphere.R,1000)
    us = sphere.rad_colloc(tind,rs)

    plt.clf()
    for l in range(0,sphere.Lsh+1):
        for m in range(-l,+l+1):
            plt.plot(sphere.rcheb,sphere.u[(l,m)][tind,:],'-o')
            plt.plot(rs,us[(l,m)],'--')
    plt.savefig(filename)

# => Running scope <= #

def u0gauss(
    x,
    y,
    z,
    ):

    return (5.0 / np.pi)**(3.0/2.0) * np.exp(-5.0 * ((x-0.5)**2 + y**2 + z**2))

sphere = Sphere(
    u0=u0gauss,
    Ncheb=64,
    Nleb=110,
    ) 

#plot_grid(sphere)

xyzw = sphere.xyzw
y = u0gauss(x=xyzw[:,0],y=xyzw[:,1],z=xyzw[:,2])
print np.sum(xyzw[:,3]*y)


#plot_rads(sphere,0,'0.pdf')
#plot_rads(sphere,10,'10.pdf')
#plot_rads(sphere,40,'40.pdf')

#x,y = np.meshgrid(
#    np.linspace(-5.0,5.0,128),
#    np.linspace(-5.0,5.0,128),
#    indexing='ij',
#    )
#z = np.zeros_like(x)
#v = sphere.colloc(10,x,y,z)
#
#print x
#print y 
#print v
#plt.clf()
#plt.contourf(x,y,v.T,np.linspace(0.0,+.05,128),cmap='seismic')
##plt.contourf(x,y,v.T,cmap='seismic')
#plt.xlabel(r'$x$')
#plt.ylabel(r'$y$')
#plt.colorbar()
#plt.savefig('0.pdf')
#
