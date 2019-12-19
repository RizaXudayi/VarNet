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
Operator code for VarNet class to solve the time-dependent AD-PDE in 1D.

Animations of the final solution corresponding to trained NNs can be found here:
    - low Peclet number:  https://vimeo.com/328756962
    - high Peclet number: https://vimeo.com/328759472

The corresponding benchmark problem can be found in:
Abdelkader Mojtabi and Michel O Deville. One-dimensional linear Advection-Diffusion equation:
Analytical and Finite Element solutions. Computers & Fluids, 107:189–195, 2015.

"""

#%% Modules:

import os

import numpy as np
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
shape = np.shape
reshape = np.reshape

from Domain import Domain1D
from ContourPlot import ContourPlot
from ADPDE import ADPDE
from VarNet import VarNet
from UtilityFunc import UF
uf = UF()

import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#%% PDE input data:

u = 1.0         # velocity magnitude
D = 0.1/pi      # diffusivity
T = 2.0         # maximum simulation time

# Initial condition:
def IC(x):
    return -sin(pi*x)

def cExact(x, t, trunc = 800):
    """
    Function to compute the analytical solution as a Fourier series expansion.
    Inputs:
        x: column vector of locations
        t: column vector of times
        trunc: truncation number of Fourier bases
    """
    # Initial condition:
    ind0 = t==0
    cInit = IC(x[ind0])
    
    # Series index:
    p = np.arange(0, trunc+1.0)
    p = reshape(p, [1, trunc+1])
    
    c0 = 16*pi**2*D**3*u*exp(u/D/2*(x-u*t/2))                           # constant
    
    c1_n = (-1)**p*2*p*sin(p*pi*x)*exp(-D*p**2*pi**2*t)                 # numerator of first component
    c1_d = u**4 + 8*(u*pi*D)**2*(p**2+1) + 16*(pi*D)**4*(p**2-1)**2     # denominator of first component
    c1 = np.sinh(u/D/2)*np.sum(c1_n/c1_d, axis=-1, keepdims=True)       # first component of the solution
    
    c2_n = (-1)**p*(2*p+1)*cos((p+0.5)*pi*x)*exp(-D*(2*p+1)**2*pi**2*t/4)
    c2_d = u**4 + (u*pi*D)**2*(8*p**2+8*p+10) + (pi*D)**4*(4*p**2+4*p-3)**2
    c2 = np.cosh(u/D/2)*np.sum(c2_n/c2_d, axis=-1, keepdims=True)       # second component of the solution
    
    # Output:
    c = c0*(c1+c2)
    c[ind0] = cInit
    
    return c


# Exact solution values for D=0.01/pi:
if D==0.01/pi:
    xEx1 = np.array([[0.9, 0.94, 0.96, 0.98, 0.99, 0.999, 1.0]]).T
    tEx1 = np.array([[0.8]])
    inpEx1 = uf.pairMats(xEx1, tEx1)
    cEx1 = np.array([[-0.30516, -0.42046, -0.47574, -0.52913, -0.55393, -0.26693, 0.0]]).T
    plt.plot(xEx1, cEx1)
    
    xEx2 = np.array([[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.94, 0.98, 0.99, 0.999, 1.0]]).T
    tEx2 = np.array([[1.0]])
    inpEx2 = uf.pairMats(xEx2, tEx2)
    cEx2 = np.array([[0.93623, 0.98441, 0.93623, 0.79641, 0.57862, 0.30420, 0.18446, 0.06181, 0.03098, 0.00474, 0.0]]).T
    plt.plot(xEx2, cEx2)
    
    xEx3 = xEx1
    tEx3 = np.array([[1.6]])
    inpEx3 = uf.pairMats(xEx3, tEx3)
    cEx3 = np.array([[0.78894, 0.85456, 0.88237, 0.90670, 0.91578, 0.43121, 0.0]]).T
    plt.plot(xEx3, cEx3)
    plt.xlabel('x')
    plt.ylabel('cEx')
    plt.grid(True)
    plt.legend(['t=0.8', 't=1.0', 't=1.6'])
    plt.show()
    
    inpEx = np.vstack([inpEx1, inpEx2, inpEx3])
    cEx = np.vstack([cEx1, cEx2, cEx3])
    
    cExact = None       # disable cExact for advective case


#%% PDE definition:

domain = Domain1D()         # 1D domain

# Setup the AD-PDE:
ADPDE_1dt = ADPDE(domain, diff=D, vel=u, timeDependent=True, tInterval=[0,T], IC=IC,
                  cEx=cExact)
#ADPDE_1dt.plotIC()
#ADPDE_1dt.plotBC(bInd=0)
#ADPDE_1dt.plotField('cEx')

#%% Architecture and discretization:

VarNet_1dt = VarNet(ADPDE_1dt,
                    layerWidth=[20],
                    discNum=20,
                    bDiscNum=None,
                    tDiscNum=300,
                    processors='GPU:0')

#%% Training:

# Folder to Store Checkpoints:
#folderpath = '/Users/Riza/Documents/Python/TF_checkpoints'
folderpath = '/home/reza/Documents/Python/TF_checkpoints'           # Linux
uf.clearFolder(folderpath)
uf.copyFile('Operator_1Dt.py', folderpath)                          # backup current operator settings

VarNet_1dt.train(folderpath, weight=[1.e1, 1.e1, 1.], smpScheme='optimal', adjustWeight=True)

#%% Simulation results:

VarNet_1dt.loadModel()
VarNet_1dt.simRes()

# Solution error:
if D==0.1/pi:
    cEx = VarNet_1dt.fixData.cEx
    cApp = VarNet_1dt.evaluate()
else:
    cApp = VarNet_1dt.evaluate(x=inpEx[:,0:1], t=inpEx[:,1:2])

string = '\n==========================================================\n'
string += 'Simulation results:\n\n'
string += 'Normalized approximation error: %2.5f' % uf.l2Err(cEx, cApp)
print(string)
VarNet_1dt.trainRes.writeComment(string)

#%% Plots for the paper:

# Load the model:
#folderpath2 = '/Users/Riza/Documents/Python/TF_checkpoints2'
#VarNet_1dt.loadModel(folderpath=folderpath, oldpath=folderpath2)
#
## Generate iteration plots:
#VarNet_1dt.trainRes.iterPlot(pltFrmt='eps')
#
## Generate solution snapshots:
#VarNet_1dt.simRes(tcoord=[0.8, 1.0, 1.6], pltFrmt='eps')

#%% Generate a movie:

# Movie of the approximate solution only:
#contPlt = ContourPlot(domain, tInterval=[0,T])      # contour plot object
#func = lambda x,t: VarNet_1dt.evaluate(x,t)         # function handle to evaluate the NN
#plotpath = VarNet_1dt.trainRes.plotpath
#
#time = np.linspace(0, T, num=101)                   # time snapshots
#for t in time:
#    contPlt.snap1Dt(func, t=t, title='t={0:.2f}s'.format(t), lineOpt='r')
#    plt.ylabel('solution')
#    plt.ylim([-1.05, 1.05])
#    filename = 't={0:.2f}s'.format(t) + '.png'
#    filepath = os.path.join(plotpath, filename)
#    plt.savefig(filepath, dpi=300)
#    plt.show()
#
#
## Movie of the exact and approximate solutions together:
#VarNet_1dt.simRes(tcoord=time)


























