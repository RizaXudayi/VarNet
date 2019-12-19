# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 17:04:47 2018

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
This file provides the classes for definition of an Advection-Diffusion PDE.

"""

#%% Modules:

import numpy as np
shape = np.shape
reshape = np.reshape
size = np.size
import numpy.linalg as la

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Domain import Domain1D
from ContourPlot import ContourPlot

from UtilityFunc import UF
uf = UF()

#%% PDE class:

class ADPDE():
    """
    Class for Advection-Diffusion PDE given as
    dc_dt = \nabla diff \nabla c - vel \cdot \nabla c + s,
    subject to
    a*\nabla c \cdot n + b*c = g on \Gamma_i.
    """
    def __init__(self,
                 domain,
                 diff,
                 vel,
                 source = 0.0,
                 timeDependent = False,
                 tInterval = None,
                 BCs = None,
                 IC = None,
                 cEx = None,
                 MORvar = None,
                 d_diff = None):
        """
        Function to initialize the AD PDE class. The AD-PDE is given as
        dc_dt = \nabla diff \nabla c - vel \cdot \nabla c + s,
        subject to
        a*\nabla c \cdot n + b*c = g on \Gamma_i.
        
        Inputs:
            domain: Domain class instance containing domain information and 
                discretization data
            diff: diffusivity field given as a constant or function of (x,t)
            vel: velocity VECTOR field given as a constant or function of (x,t)
            source: source field given as a function of (x,t)
            tInterval [1x2]: time interval 
            BCs: list containing [a, b, g(x,t)] corresponding to each boundary
                indicator (default BC is zero-value Dirichlet)
            BCtype: list of corresponding boundary condition types:
                'Dirichlet', 'Neumann', 'Robin'
            IC: initial condition for time-dependent problem (default: zero)
            cEx: exact concentration field given as a function of (x,t)
            MORvar: an instance of MOR class containing the variable arguments 
                for the input data diff, vel, source, IC, and BCs (default: empty)
            d_diff: gradient of the diffusivity field wrt spatial coordinate
            
        Note that all function handles should recieve column vectors (x,t) in 
        that order and return a column vector of the same length.
        
        Attributes:
            dim: dimension of the problem
            timeDependent: boolean
            domain: instance of Domain class over which the PDE is solved
            diff, diffFun: diffusivity field
            vel, velFun: velocity vector field
        """
        if uf.isnone(tInterval):    timeDependent = False
        else:                       timeDependent = True
        
        if not uf.isnumber(diff) and not callable(diff):
            raise ValueError('diffusivity field must be constant or callable!')
            
        if not uf.isnumber(vel) and not callable(vel):
            raise ValueError('velocity field must be constant or callable!')
            
        if not uf.isnumber(source) and not callable(source):
            raise ValueError('source function must be constant or callable!')
            
        if not uf.isnone(BCs) and type(BCs) is not list:
            raise ValueError('BCs must be empty or a list of [a, b, g(x,t)]!')
        elif not uf.isnone(BCs) and len(BCs)!=domain.bIndNum:
            raise ValueError('number of BCs does not match number of boundaries in domain!')
            
        if timeDependent and uf.isnone(IC):
            raise ValueError('initial condition must be provided for time-dependent problems!')
            
        if not uf.isnone(cEx) and not callable(cEx):
            raise ValueError('exact solution must be a callable function!')
        
        if not uf.isnone(d_diff) and not uf.isnumber(d_diff) and not callable(d_diff) :
            raise ValueError('diffusivity gradient must be constant or callable!')
            
        
        dim = domain.dim            # domain dimension
        
        # Define callable functions for constant fields:    
        if not callable(diff):
            diffFun = lambda x, t=0: diff*np.ones([shape(x)[0], 1])
            self.diff = diff
            self.diffFun = diffFun
        else:
            diffFun = diff
            self.diffFun = diffFun
        
        if not callable(vel):
            velFun = lambda x, t=0: vel*np.ones([shape(x)[0], dim])
            self.vel = vel
            self.velFun = velFun
        else:
            velFun = vel
            self.velFun = velFun
        
        if not callable(source):
            sourceFun = lambda x, t=0: source*np.ones([shape(x)[0], 1])
            self.source = source
            self.sourceFun = sourceFun
        else:
            sourceFun = source
            self.sourceFun = sourceFun
        
        if not callable(d_diff):
            if uf.isnone(d_diff):
                d_diff = 0.0
            d_diffFun = lambda x, t=0: d_diff*np.ones([shape(x)[0], dim])
            self.d_diff = d_diff
            self.d_diffFun = d_diffFun
        else:
            self.d_diffFun = d_diff
        
        # Process the BCs and set them into standard format:
        bIndNum = domain.bIndNum                        # number of boundary indicatros
        if uf.isnone(BCs):                              # allocate a list if BCs==[]
            BCs = [[] for i in range(bIndNum)]
        for bInd in range(bIndNum):                     # loop over BCs
            if uf.isempty(BCs[bInd]):                   # if empty, define Dirichlet BC
                BCs[bInd] = [0.0, 1.0, lambda x, t=0: np.zeros([len(x), 1])]
            elif len(BCs[bInd])!=3:
                raise ValueError('BCs must be specified as a list of [a, b, g(x,t)]!')
            elif not callable(BCs[bInd][2]):
                BCval = BCs[bInd]                       # if not coppied as a whole, acts as pointer and generates errors
                BCs[bInd] = [ BCval[0], BCval[1], lambda x, t=0: BCval[2]*np.ones([len(x), 1]) ]
                
        BCtype = []
        for bInd in range(bIndNum):                     # loop over BCs
            if BCs[bInd][0]==0:
                BCtype.append('Dirichlet')
            elif BCs[bInd][1]==0:
                BCtype.append('Neumann')
            else:
                BCtype.append('Robin')
            
        # Process the initial condition:
        if timeDependent and uf.isempty(IC):
            IC = lambda x, t=0: np.zeros([len(x), 1])
        elif timeDependent and not callable(IC):
            ICval = IC
            IC = lambda x, t=0: ICval*np.ones([len(x), 1])
            
        # Check PDE input data for variable arguments and create a lookup table:
        if not uf.isnone(MORvar):
            funcHandles = MORvar.funcHandles            # list of function handles
            funNum = MORvar.funNum                      # number of function handles
            bDataFlg = False                            # flag to determine if boundary data appears in MOR
            BCind = [None for i in range(bIndNum)]
            MORfunInd = {'diff':None, 'vel':None, 'source':None, 'IC':None, 'd_diff':None}
            for i in range(funNum):
                if funcHandles[i]==diffFun:
                    MORfunInd['diff'] = i
                elif funcHandles[i]==velFun:
                    MORfunInd['vel'] = i
                elif funcHandles[i]==sourceFun:
                    MORfunInd['source'] = i
                elif funcHandles[i]==IC:
                    MORfunInd['IC'] = i
                elif funcHandles[i]==d_diffFun:
                    MORfunInd['d_diff'] = i
                else:
                    for bInd in range(bIndNum):
                        if funcHandles[i]==BCs[bInd][2]:
                            BCind[bInd] = i
                            bDataFlg = True
                            
            if not uf.isnone(MORfunInd['diff']) and callable(d_diff) and uf.isnone(MORfunInd['d_diff']):
                raise ValueError('\'diff\' has extra input arguments but \'d_diff\' does not!')
                
            MORfunInd['BCs'] = BCind
            if uf.isnone(MORfunInd['diff']) and uf.isnone(MORfunInd['vel']) and uf.isnone(MORfunInd['source']):
                MORfunInd['inpData'] = None
            else:
                MORfunInd['inpData'] = True
            if bDataFlg or not uf.isnone(MORfunInd['IC']):
                MORfunInd['biData'] = True
            else:
                MORfunInd['biData'] = None
            self.MORfunInd = MORfunInd
            
        self.dim = dim
        self.domain = domain
        self.timeDependent = timeDependent
        self.tInterval = tInterval
        self.BCs = BCs
        self.BCtype = BCtype
        self.IC = IC
        self.cEx = cEx
        self.MORvar = MORvar
        
        
    def plotIC(self):
        """
        Function to plot the initial condition for time-dependent problems.
        """
        dim = self.dim
        domain = self.domain
        
        if not self.timeDependent:
            raise TypeError('the PDE is time-independent!')
        
        if dim==1:
            mesh = domain.getMesh()
            coord = mesh.coordinates
            plt.figure()
            plt.plot(coord, self.IC(coord))
            plt.xlabel('$x$')
            plt.title('initial condition')
            plt.grid(True)
            plt.show()
            
        elif dim==2:
            contPlot = ContourPlot(domain)
            contPlot.conPlot(self.IC)
            
            
    def plotBC(self, bInd):
        """
        Function to plot the right-hand-side of the BC specified by 'bInd'.
        
        Input:
            bInd: number of the boundary indicator to be plotted
        """
        dim = self.dim
        BCs = self.BCs
        domain = self.domain
        tInterval = self.tInterval
        
        if bInd>=domain.bIndNum:
            raise ValueError('entered boundary index is larger than number of indicators!')
            
        g = BCs[bInd][2]                                            # function handle to rhs of BC
        
        if dim==1 and not self.timeDependent:
            raise TypeError('boundary conditions can only be plotted for time-dependent 1D problems!')
        elif dim==1:
            time = np.linspace(tInterval[0], tInterval[1])
            plt.figure()
            plt.plot(time, g(time))
            plt.xlabel('time')
            plt.ylabel('$g(x,t)$')
            plt.title('boundary condition {}'.format(bInd+1))
            plt.grid(True)
            
        elif dim==2 and not self.timeDependent:
            mesh = domain.getMesh()
            coord = mesh.bCoordinates[bInd]                         # discretized coordinates of the boundary
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot3D(coord[:,0], coord[:,1], g(coord)[:,0])
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            ax.set_zlabel('$g(x)$')
            plt.title('boundary condition {}'.format(bInd+1))
            plt.grid(True)
            
        elif dim==2:
            bg = domain.boundryGeom[bInd, :,:]                      # limits for current boundary edge
            domain0 = Domain1D(interval=[0,1])
            contPlot = ContourPlot(domain0, tInterval)
            func = lambda s,t: g(bg[0,:] + s*(bg[1,:]-bg[0,:]),t)   # map to 2D domain
            contPlot.conPlot(func, title='boundary condition {}'.format(bInd+1))
            plt.xlabel('unit step along boundary {}'.format(bInd+1))
            plt.ylabel('$t$')
            print('\nstep direction from ' + str(bg[0,:]) + ' to ' + str(bg[1,:]) + ':')
        
        plt.show()
        
        
    def plotField(self, fieldName, t=[]):
        """
        Function to plot the fields corresponding to input data for the PDE.
        
        Input:
            fieldName: name of the field:
                'diff': diffusivity
                'vel': velocity
                'source': source
                'cEx': exact solution
        """
        dim = self.dim
        domain = self.domain
        timeDependent = self.timeDependent
        tInterval = self.tInterval
        
        if fieldName=='diff':
            field = self.diffFun;       fieldname = 'diffusivity field'
            
        elif fieldName=='vel':
            velFun = self.velFun
            fieldname = 'velocity field'
            if dim==1:
                field = velFun
            elif dim==2 and not timeDependent:
                field = lambda x: la.norm(velFun(x), axis=1)
            elif dim==2:
                field = lambda x,t: la.norm(velFun(x,t), axis=1)
                
        elif fieldName=='source':
            field = self.sourceFun;     fieldname = 'source field'
            
        elif fieldName=='cEx':
            cEx = self.cEx
            if uf.isnone(cEx): raise ValueError('exact solution function not provided!')
            field = cEx;       fieldname = 'exact solution field'
            
        else:
            raise ValueError('incorrect field name!')
            
        if dim==1:
            contPlot = ContourPlot(domain, tInterval)
            contPlot.conPlot(field, title=fieldname)
            
        elif dim==2 and not timeDependent:
            contPlot = ContourPlot(domain)
            contPlot.conPlot(field, title=fieldname)
            
        elif dim==2:
            contPlot = ContourPlot(domain, tInterval)
            contPlot.conPlot(field, t, title=fieldname)
                
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        