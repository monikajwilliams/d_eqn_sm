#!/usr/bin/env python -u
import os,sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import util
import numpy as np
import matplotlib.pyplot as plt

class Sphere(object):

    def __init__(
        self,
        u0=None, #function to evaluate initial density as fxn of x,y,z
        u1=None, #potential of mean force as a fxn of x,y,z
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

        self.Ri = Ri
        self.Ro = Ro
        self.R = self.Ro-self.Ri
        self.k = k
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
        # chebyshev nodes mapped to r
        self.rcheb = (self.Ro - self.Ri)/2.0 * (self.xcheb + 1.0) + self.Ri
        # chebyshev radal weights [Ri, Ro]
        self.vcheb = self.wcheb * self.R/2.0 * self.rcheb**2
   
        # time slices 
        self.ts = np.arange(0.0,self.tmax + 0.5*self.dt,self.dt)

        # state vector initialization
        self.u = {}
        self.uf = {}
        for l in range(0,self.Lsh+1):
            for m in range(-l,l+1):
                self.u[(l,m)] = np.zeros((len(self.ts),self.Ncheb+1))
                self.uf[(l,m)] = np.zeros((len(self.ts),self.Ncheb+1))

        # initial conditions
        theta = self.leb.theta 
        phi = self.leb.phi 
        wleb = self.leb.w
        Y = util.sh(theta,phi,self.Lsh)
        ur = np.zeros(len(self.rcheb))
        ru = np.zeros(len(self.rcheb))
        for rind,r in enumerate(self.rcheb):
            x,y,z = util.sphere_to_cart(r,theta,phi) 
            u0val = u0(x,y,z,self.Ro,self.Ri)
            ufval = u1(x,y,z,self.Ro,self.Ri)
            #saving radial positions and energies for test plotting
            ru[rind] =  np.sqrt(x**2 + y**2 + z**2)[0]
            ur[rind] = ufval[0]
            for l in range(0,self.Lsh+1):
                for m in range(-l,l+1):
                    Yval = Y[(l,m)]
                    self.u[(l,m)][0,rind] = np.sum(wleb*Yval*u0val)
                    self.uf[(l,m)][0,rind] = np.sum(wleb*Yval*ufval)

        # establish the propagator (Crank Nicolson)
        self.a = self.v*self.dt/2.0
        I = np.eye(self.Ncheb+1)
        D = util.cheb_D(self.Ncheb)
        M = 2.0/self.R*D
        K0 = np.dot(np.diag(1.0/self.rcheb**2),np.dot(M,np.dot(np.diag(self.rcheb**2),M)))

        # calculate the force
        K1 = D

        K = {}
        for l in range(0,self.Lsh+1):
            K[l] = K0 - np.diag(l*(l+1.0)/self.rcheb**2)
        self.U = {}
        for l in range(0,self.Lsh+1):
            L = I - self.a*K[l]
            R = I + self.a*K[l]
            # No flux on periphery
            L[0,:] = M[0,:]
            R[0,:] = 0.0
            if l != 0:
                # No signal at origin for l != 0
                L[-1,:] = 0.0
                L[-1,-1] = 1.0
                R[-1,:] = 0.0
            else:
                # Reactive BC at origin for l == 0
                L[-1,:] = M[-1,:]
                R[-1,:] = 0.0
                R[-1,-1] = self.k 
            self.U[l] = np.linalg.solve(L,R)

        # Propagate!
        for l in range(0,self.Lsh+1):
            for m in range(-l,l+1):
                #for ind in range(1,len(self.ts)):
                for ind in range(1,2):
                    self.u[(l,m)][ind,:] = np.einsum('ij,j->i',self.U[l],self.u[(l,m)][ind-1,:])
                    self.uf[(l,m)][ind,:] = -np.dot(K1[l],self.uf[(l,m)][ind-1,:])
                    print self.uf[(l,m)][ind,:]

        tind=1
        us = self.rad_colloc(tind,ru)
        plt.clf()
        for l in range(0,self.Lsh+1):
            for m in range(-l,+l+1):
                plt.plot(self.rcheb,self.uf[(l,m)][tind,:],'-o')
                plt.plot(ru,us[(l,m)],'--')

        plt.plot(ru,ur)
        plt.plot(ru,-1.0/(ru**2))
        plt.show()
        exit()
        
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

        x = (r - self.Ri)*(2.0/(self.Ro-self.Ri)) - 1.0 
        args = np.where((x >= -1.0) & (x<= 1.0))

        vals = {}  
        for l in range(0,self.Lsh+1):
            for m in range(-l,l+1):
                #avals = util.cheb_trans2(self.uf[(l,m)][tind,:],self.Ncheb)
                avals = util.cheb_trans2(self.uf[(l,m)][tind,:],self.Ncheb)
                vals[(l,m)] = np.zeros_like(x)
                vals[(l,m)][args] += util.cheb_colloc(x[args],avals)

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
    filename,
    ):

    xyzw = sphere.xyzw
    xs = xyzw[:,0]
    ys = xyzw[:,1]
    zs = xyzw[:,2]
    ws = xyzw[:,3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=ws)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(filename)

def plot_rads(
    sphere,
    tind,
    filename,
    ):

    rs = np.linspace(0.0,sphere.Ro,1000)
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

# => Homework Scope <= #

def plot_basis(filename):

    Ro = 8.0
    Ri = 3.0
    sphere = Sphere(
        k=8.5,
        u0=u0annulus,
        Ncheb=64,
        Nleb=110,
        tmax=50.0,
        dt = 0.1,
        Ri=Ri, 
        Ro=Ro, 
        v = 1.0,
        ) 
    
    plot_grid(sphere,filename)

def plot_integral(
    filename,
    Ro = 10.0,
    Ri = 0.5,
    k=8.5,
    tmax=100.0,
    t_interval=5.0,
    ):

    sphere = Sphere(
        k=k,
        u0=u0annulus,
        Ncheb=64,
        Nleb=110,
        tmax=tmax,
        dt = 0.1,
        Ri=Ri, 
        Ro=Ro, 
        v = 1.0,
        ) 

    w = sphere.xyzw[:,3]
    x = sphere.xyzw[:,0]
    y = sphere.xyzw[:,1]
    z = sphere.xyzw[:,2]
    ts = np.arange(0,tmax,t_interval)
    integral = np.zeros_like(ts)
    for ind,t in enumerate(ts):
        v = sphere.colloc(t,x,y,z)
        integral[ind] = np.sum(w*v)
        if ind == 0:
            norm = integral[ind]
        integral[ind] /= norm

    plt.clf()
    plt.plot(ts,integral,'-o')
    plt.xlabel(r'$\mathrm{t}$',fontsize=18)
    plt.ylabel(r'$\mathrm{Density\ Integral}$',fontsize=18)
    plt.axis([0.0,tmax,0.0,1.5])
    plt.savefig(filename)

def make_gifs(
    Ro = 10.0,
    Ri = 1.0,
    k=8.5,
    tmax=5000.0,
    log=True,
    ):

    sphere = Sphere(
        k=k,
        u0=u0annulus,
        u1=u0r,
        Ncheb=64,
        Nleb=110,
        tmax=tmax,
        dt = 0.1,
        Ri=Ri, 
        Ro=Ro, 
        v = 1.0,
        ) 
    
    t1 = np.arange(0.0,5.0,0.5)
    t2 = np.arange(5.0,10.0,1.0)
    t3 = np.arange(300,tmax+50.0,50.0)
    t_plots = np.hstack((t1,t2,t3))
    print "nplots = %d" % (len(t_plots))
    os.system('mkdir gif_k%d/' % (int(k)))

    v0 = sphere.colloc(0,sphere.xyzw[:,0],sphere.xyzw[:,1],sphere.xyzw[:,2])
    w = sphere.xyzw[:,3]
    norm = np.sum(v0*w)
    top = np.max(v0/norm)

    if log == False:
        lvls = np.linspace(0.0,top,500)
        interval = np.arange(0,len(lvls),50)  
        labels = []
        for label in lvls[interval]:
            labels.append('%1.1f' % (label))
    else:
        lvls = np.logspace(-6.0,np.log10(top)+0.5,500)
        interval = np.arange(0,len(lvls),50)  
        labels = []
        for label in np.log10(lvls[interval]):
            labels.append('%1.1f' % (label))
    xlin = np.linspace(-Ro-1.0,Ro+1.0,128)
    ylin1 = np.sqrt(Ro**2 - xlin**2)
    ylin2 = np.sqrt(Ri**2 - xlin**2)

    for t in t_plots:
    
        x,y = np.meshgrid(
            np.linspace(-Ro-1.0,Ro+1.0,128),
            np.linspace(-Ro-1.0,Ro+1.0,128),
            indexing='ij',
            )
        z = np.zeros_like(x)
        v = sphere.colloc(t,x,y,z)/norm
        
        print "<=========>"
        print t
        print np.max(v)
        print np.min(v)
    
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
   
        print "plotting ring" 
        a1 = ax.plot(xlin, ylin1,'--k',alpha=0.5)
        a2 = ax.plot(xlin,-ylin1,'--k',alpha=0.5)
        a3 = ax.plot(xlin, ylin2,'--k',alpha=0.5)
        a4 = ax.plot(xlin,-ylin2,'--k',alpha=0.5)

        print "plotting contour" 
        if log == False:     
            axim = ax.contourf(x,y,v.T,levels=lvls,cmap='Blues')
        else:
            axim = ax.contourf(x,y,np.abs(v.T),levels=lvls,cmap='Blues',norm=LogNorm())
    
        print "saving figure" 
        cb = fig.colorbar(axim,ticks=lvls[interval])
        cb.ax.set_yticklabels(labels)
        plt.xlabel(r'$x$',fontsize=18)
        plt.ylabel(r'$y$',fontsize=18)
        plt.savefig('gif_k%d/%07.1f_abs.png'%(int(k),t))
        plt.close()
    os.system('/usr/local/bin/convert -delay 20 -loop 0 "gif_k%d/*.png" k%d.gif' % (int(k),int(k)))
    
#plot_basis('an_basis.pdf') 
#plot_integral('abs_int.pdf',tmax=5000.0,t_interval=100.0,k=8.5) 
#plot_integral('ref_int.pdf',tmax=5000.0,t_interval=100.0,k=0.0) 
#make_gifs(tmax=5000.0,k=8.0)
make_gifs(tmax=2.0,k=0.0)
    
