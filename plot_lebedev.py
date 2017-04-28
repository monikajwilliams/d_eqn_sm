#!/usr/bin/env python
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import util
import sys

def plot_lebedev(
    N,
    ):

    leb = util.Lebedev.build(N)
    print leb.order
    print leb
    xyzw = leb.xyzw
    x = xyzw[:,0]
    y = xyzw[:,1]
    z = xyzw[:,2]
    w = xyzw[:,3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=w)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('lebedev_grid.pdf')

plot_lebedev(int(sys.argv[1]))

     
