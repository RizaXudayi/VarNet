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
Operator code for VarNet class to solve the parametric time-dependent AD-PDE in 1D.

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
from MOR import MOR
from VarNet import VarNet
from UtilityFunc import UF
uf = UF()

import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#%% PDE input data:

u = 1.0         # velocity magnitude
T = 2.0         # maximum simulation time

def diffFun(x, t=0, D=0.1/pi):
    """Diffusivity field function."""
    return D*np.ones([shape(x)[0], 1])

def IC(x):
    """Initial condition."""
    return -sin(pi*x)

def cExact(x, t, D=0.1, trunc=800):
    """
    Function to compute the analytical solution as a Fourier series expansion.
    Inputs:
        x: column vector of locations
        t: column vector of times
        D: diffusivity value
        trunc: truncation number of Fourier bases
    """
    if D<0.1/pi:
        raise ValueError('analytical solution is unstable for requested diffusivity value!')
        
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


# Exact solution values for the cases that the analytical solution is unstable:
# Accuracy points for D = 0.01/pi:
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
plt.title('$D = 0.01/\pi$')
plt.xlabel('$x$')
plt.ylabel('cEx')
plt.grid(True)
plt.legend(['t=0.8', 't=1.0', 't=1.6'])
plt.show()

inpEx = np.vstack([inpEx1, inpEx2, inpEx3])
cExD3 = np.vstack([cEx1, cEx2, cEx3])


# Accuracy points for D = 5e-3:
cEx1 = np.array([[-0.29706, -0.40929, -0.46288, -0.50386, -0.46059, -0.09798, 0.0]]).T
plt.plot(xEx1, cEx1)

cEx2 = np.array([[0.90527, 0.95185, 0.90526, 0.77006, 0.55948, 0.29414, 0.17836, 0.06086, 0.03394, 0.00544, 0.0]]).T
plt.plot(xEx2, cEx2)

cEx3 = np.array([[0.74874, 0.81004, 0.83652, 0.81925, 0.68929, 0.22293, 0.0]]).T
plt.plot(xEx3, cEx3)
plt.title('$D = 0.005$')
plt.xlabel('$x$')
plt.ylabel('cEx')
plt.grid(True)
plt.legend(['t=0.8', 't=1.0', 't=1.6'])
plt.show()

cExD4 = np.vstack([cEx1, cEx2, cEx3])


#%% MOR class:

def discDiff(discNum=6):
    """Function to discretize the diffusivity value."""
    return np.array([0.003*(11**(n/(discNum-1))) for n in range(discNum)])[np.newaxis].T

funcHandles = diffFun
ArgNames = ['D']
ArgRange = [[0.003, 0.033]]
discScheme = discDiff

MORvar = MOR(funcHandles, ArgNames, ArgRange)

#%% PDE definition:

domain = Domain1D()         # 1D domain

ADPDE_1dt = ADPDE(domain, diff=diffFun, vel=u, timeDependent=True,
                  tInterval=[0,T], IC=IC, MORvar=MORvar)
#ADPDE_1dt.plotIC()
#ADPDE_1dt.plotBC(bInd=0)

#%% Architecture and discretization:

VarNet_1dt = VarNet(ADPDE_1dt,
                    layerWidth=[10, 20, 30],
                    discNum=150,
                    bDiscNum=75,
                    tDiscNum=800,
                    MORdiscScheme=discScheme,
                    processors='GPU:1')

#%% Training:

# Folder to Store Checkpoints:
#folderpath = '/Users/Riza/Documents/Python/TF_checkpoints'
folderpath = '/home/reza/Documents/Python/TF_checkpoints2'           # Linux
uf.clearFolder(folderpath)
uf.copyFile('Operator_1DtMOR.py', folderpath)                        # backup current operator settings

VarNet_1dt.train(folderpath, weight=[1.e1, 1.e1, 1.], smpScheme='uniform', saveMORdata=True, batchNum=20, shuffleData=True)

#%% Simulation results:

VarNet_1dt.loadModel()

# Solution error:
errVal = []
tcoord = reshape(np.linspace(0, T, num=100), [100,1])               # temporal discretization
coord = domain.getMesh(discNum=[100], bDiscNum=50).coordinates      # spatial discretization
Input = uf.pairMats(coord, tcoord)

kapa = 0.01/pi
VarNet_1dt.simRes(batch=0)
cApp = VarNet_1dt.evaluate(x=inpEx[:,0:1], t=inpEx[:,1:2], MORarg=[[kapa]])
errVal.append(uf.l2Err(cExD4, cApp))

kapa = 0.005
VarNet_1dt.simRes(batch=1)
cApp = VarNet_1dt.evaluate(x=inpEx[:,0:1], t=inpEx[:,1:2], MORarg=[[kapa]])
errVal.append(uf.l2Err(cExD3, cApp))

kapa = 0.1/pi
VarNet_1dt.simRes(batch=5)
cEx = cExact(x=Input[:,:1], t=Input[:,1:2], D=kapa)
cApp = VarNet_1dt.evaluate(x=Input[:,:1], t=Input[:,1:2], MORarg=[[kapa]])
errVal.append(uf.l2Err(cEx, cApp))

kapa = 0.1
cEx = cExact(x=Input[:,:1], t=Input[:,1:2], D=kapa)
cApp = VarNet_1dt.evaluate(x=Input[:,:1], t=Input[:,1:2], MORarg=[[kapa]])
errVal.append(uf.l2Err(cEx, cApp))

string = '\n==========================================================\n'
string += 'Simulation results:\n\n'
string += 'Normalized approximation error for D = %2.5f: %2.5f\n' % (0.01/pi, errVal[0])
string += 'Normalized approximation error for D = %2.5f: %2.5f\n' % (0.005,   errVal[1])
string += 'Normalized approximation error for D = %2.5f: %2.5f\n' % (0.1/pi,  errVal[2])
string += 'Normalized approximation error for D = %2.5f: %2.5f'   % (0.1,     errVal[3])
print(string)
VarNet_1dt.trainRes.writeComment(string)

#%% Plots for the paper:

## Load the model:
#folderpath = '/Users/Riza/Documents/Python/TF_checkpoints'
#VarNet_1dt.loadModel(folderpath=folderpath)
#
## Generate iteration plots:
#VarNet_1dt.trainRes.iterPlot(pltFrmt='eps')
#
## Generate solution snapshots:
#VarNet_1dt.simRes(tcoord=[0.8, 1.0, 1.6], pltFrmt='eps')

## Overly the available exact solution on the solution:
#if D==0.01/pi:
#    tcoord = [0.8, 1.0, 1.6]
#    contPlot = ContourPlot(domain, tInterval=[0,T])
#    cAppFun = lambda x, t: VarNet_1dt.evaluate(x, t)        # function handles
#    col = 'brk'
#    Legend = []
#    plotpath = os.path.join(folderpath, 'plots')
#    
#    for i in range(3):
#        t = tcoord[i]
#        contPlot.snap1Dt(cAppFun, t, figNum=1, lineOpt=col[i])
#        Legend.append('t={0:.2f}s'.format(t))
#        
#    for i in range(3):
#        t = tcoord[i]
#        if t==0.8:
#            plt.plot(inpEx1[:,0:1], cEx1, col[i] + '.')
#        elif t==1.0:
#            plt.plot(inpEx2[:,0:1], cEx2, col[i] + '.')
#        else:
#            plt.plot(inpEx3[:,0:1], cEx3, col[i] + '.')
#            
#    plt.legend(Legend)
#    plt.ylabel('solution')
#    filepath = os.path.join(plotpath, 'cApp2.png')
#    plt.savefig(filepath, dpi=300)
#    filepath = os.path.join(plotpath, 'cApp2.eps')
#    plt.savefig(filepath, dpi=300)
#    plt.show()


#%% Generate a movie:

## Load the model:
#folderpath2 = '/home/reza/Documents/Python/TF_checkpoints'
#VarNet_1dt.loadModel(folderpath=folderpath, oldpath=folderpath2)
#
## Movie of the approximate solution only:
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

#%% Parametric solution:

## Load the model:
#folderpath = '/Users/Riza/Documents/Python/TF_checkpoints'
#VarNet_1dt.loadModel(folderpath=folderpath)

#t = 1.5
#plotpath = VarNet_1dt.trainRes.plotpath
#coord = domain.meshTot(discNum=[100]).coordinates      # spatial discretization
##kapaVals = discDiff()
#kapaVals = np.linspace(0.003, 0.033, num=10)[np.newaxis].T
#Legend = []
#for kapa in kapaVals:
#    cApp = VarNet_1dt.evaluate(x=coord, t=t, MORarg=[kapa])
#    plt.plot(coord, cApp)
#    Legend.append('$\kappa = %2.4f$' % kapa)
#plt.xlabel('$x$')
#plt.ylabel('solution')
#plt.legend(Legend)
#plt.grid(True)
#filepath = os.path.join(plotpath, 'paramSol.eps')
#plt.savefig(filepath, dpi=300)
#plt.title('solution at $t=%2.1f$ as a function of $\kappa$' % t)
#filepath = os.path.join(plotpath, 'paramSol.png')
#plt.savefig(filepath, dpi=300)
#plt.show()





















