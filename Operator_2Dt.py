# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 12:33:03 2018

-------------------------------------------------------------------------------
=============================== VarNet Library ================================
-------------------------------------------------------------------------------

Authors: Reza Khodayi-mehr and Michael M Zavlanos
reza.khodayi.mehr@duke.edu
http://people.duke.edu/~rk157/
Department of Mechanical Engineering and Materials Science,
Duke University, Durham, NC 27708, USA.

Copyright (c) 2019 Reza Khodayi-mehr - licensed under the MIT License
For a full copyright statement see the accompanying LICENSE.md file.
    
For theoretical derivations as well as numerical experiment results, see:
Reza Khodayi-mehr and Michael M Zavlanos. VarNet: Variational neural networks
for the solution of partial differential equations, 2019.
https://arxiv.org/pdf/1912.07443.pdf

To examine the functionalities of the VarNet library, see the acompanying 
Operater files.

The code is fully functional with the following module versions:
    - tensorflow: 1.10.0
    - numpy: 1.16.4
    - scipy: 1.2.1
    - matplotlib: 3.0.3

-------------------------------------------------------------------------------
Operator code for VarNet class to solve a 2D time-dependent AD-PDE with an 
analytical solution, reported in Leuj and Dane's paper.

An animation of the final solution corresponding to trained NN can be found
here: https://vimeo.com/350221337

The corresponding benchmark problem can be found in:

F. J. Leij and J. Dane, “Analytical solution of the one-dimensional
advection equation and two- or three-dimensional dispersion equation,”
Water Resources Research, vol. 26, pp. 1475–1482, 07 1990.

P. Siegel, R. Mose, P. Ackerer, and J. Jaffr ´ e, “Solution of the advection– ´
diffusion equation using a combination of discontinuous and mixed finite
elements,” International journal for numerical methods in fluids, vol. 24,
no. 6, pp. 595–613, 1997.
    
"""

#%% Modules:

import os

import numpy as np
shape = np.shape
reshape = np.reshape
sin = np.sin
cos = np.cos
exp = np.exp
pi = np.pi

from scipy import special
erf = special.erf                       # error function

from Domain import PolygonDomain2D
from ContourPlot import ContourPlot
from ADPDE import ADPDE
from VarNet import VarNet
from UtilityFunc import UF
uf = UF()

import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


#%% PDE input data:

#lim = np.array([[0, -0.5], [2, 0.5]])  # spatial domain boundaries
T = 1.5                                 # maximum simulation time
q = [1., 0.]                            # velocity magnitude
kappa = 1.e-3                           # diffusivity
c0 = 1.0                                # concentration value on x=0, |y|<a
a = 0.2                                 # bounds on the concentration BC
nt = 151                                # time discretization number for results

def cExFun(x, t=None):
    """
    Exact analytical solution (involves integration over temporal coordinate).
    Inputs:
        x [nx x dim]: vector of spatial discretizations
        t: vector of temporal values at which the solution is requested
    Output [nx, nt]: each row contains the values for a given spatial point
        at different times
    Note: this function cannot be used with the VarNet toolbox since it does
        not conform to its standard form and instead is optimized for faster evaluation.
    """
    # Discretization of the spatial-temporal domain:
    nx = len(x)
    tcoord = np.linspace(0, T, num=nt)  # discretize the temporal coordinate
    tcoord = tcoord[1:]                 # remove t=0 to avoid division by zero
    tcoord = reshape(tcoord, newshape=[-1,1])
    dt = T/(nt-1)                       # temporal interval
    Input = uf.pairMats(x,tcoord)       # create copies for all spatial discretization points
    x1 = Input[:,0:1]
    x2 = Input[:,1:2]
    tcoord = Input[:,2:3]
    
    # Integrand:
    qx = q[0]
    integ = c0*x1/4/np.sqrt(pi*kappa) * tcoord**(-1.5) * exp( -(x1-qx*tcoord)**2 /4/kappa/tcoord )
    denom = 1/2/np.sqrt(kappa*tcoord)
    integ = integ * ( erf( (a+x2)*denom ) + erf( (a-x2)*denom )  )
    
    # Integral over spatial coordinate:
    integ = reshape(integ, newshape=(nx, nt-1))     # reshape to get all times for each spatial point separately
    integ = dt*np.cumsum(integ, axis=1)             # integrate over temporal coordinate for all time instances
    integ = np.hstack([np.zeros([nx,1]), integ])    # append the initial condition
    
    # If the value at specific times is requested:
    tcoord = np.linspace(0, T, num=nt)
    if not uf.isnone(t):
        ind = uf.nodeNum(tcoord, t)
        integ = integ[:, ind]
    
    # Enforce the BC at x = 0 and |y| < a:
    ind = (x[:,0]<1.e-4)*(x[:,1]<a)*(-a<x[:,1])
    integ[ind,:] = c0
    
    return integ

#%% Domain definition and contour plots:

vertices = np.array([[0.0, -0.5], [0.0, -0.2], [0.0, 0.2], [0.0, 0.5], 
                     [2.0, 0.5], [2.0, -0.5]])
domain = PolygonDomain2D(vertices)

contPlt = ContourPlot(domain, tInterval=[0, T])

contPlt.animPlot(cExFun)

#%% AD-PDE:

BC = [[], [0.0, 1.0, c0], [], [], [], []]
ADPDE_2d = ADPDE(domain, diff=kappa, vel=q, tInterval=[0,T], BCs=BC, IC=0.0)
#ADPDE_2d.plotBC(1)

#%% Architecture and discretization:

VarNet_2d = VarNet(ADPDE_2d,
                    layerWidth = [10, 20],
                    discNum=[80, 40],
                    bDiscNum=40,
                    tDiscNum=75,
                    processors = 'GPU:0')

#%% Training:

# Folder to Store Checkpoints:
#folderpath = '/Users/Riza/Documents/Python/TF_checkpoints'
folderpath = '/home/reza/Documents/Python/TF_checkpoints'           # Linux
uf.clearFolder(folderpath)
uf.copyFile('Operator_anal2Dt.py', folderpath)                      # backup current operator settings

VarNet_2d.train(folderpath, weight=[5, 1, 1], smpScheme='uniform')

#%% Simulation results:

VarNet_2d.loadModel()
VarNet_2d.simRes()

# Solution error:
tcoord = reshape(np.linspace(0, T, num=nt), [nt,1])                 # temporal discretization
coord = domain.getMesh(discNum=[40,20], bDiscNum=20).coordinates    # spatial discretization
cEx = cExFun(coord)
cEx = reshape(cEx, newshape=(-1,1))
Input = uf.pairMats(coord, tcoord)
cApp = VarNet_2d.evaluate(x=Input[:,:2], t=Input[:,2:3])

string = '\n==========================================================\n'
string += 'Simulation results:\n\n'
string += 'Normalized approximation error: %2.5f' % uf.l2Err(cEx, cApp)
print(string)
VarNet_2d.trainRes.writeComment(string)

#%% Plots for the paper:

## Load the model:
##folderpath = '/Users/Riza/Documents/Python/TF_checkpoints'
#folderpath = '/home/reza/Documents/Python/TF_checkpoints2'           # Linux
#VarNet_2d.loadModel(folderpath=folderpath)
#
## Generate iteration plots:
#VarNet_2d.trainRes.iterPlot(pltFrmt='eps')
#
## Generate solution snapshots:
#VarNet_2d.simRes(tcoord=[0.5, 1.0, 1.5], pltFrmt='eps')

# Plot the concentration front along the longitudinal profile:
x1 = reshape(np.linspace(0.0, 2.0, 100), [-1,1])
coord = uf.hstack([x1, np.zeros([100,1])])
cEx = cExFun(coord, t=1.0)
cApp = VarNet_2d.evaluate(x=coord, t=1.0)
plt.plot(x1, cEx, 'b')
plt.plot(x1, cApp, 'r.')
plt.grid(True)
plt.xlabel('$x_1$')
plt.ylabel('$c(t, \mathbf{x})$')
plt.legend(['exact', 'approximate'])
filepath = os.path.join(VarNet_2d.trainRes.plotpath, 'cFront.eps')
plt.savefig(filepath, dpi=300)
plt.title('concentration for longitudinal profile at $t=1.0$')
plt.show()









