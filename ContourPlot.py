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
This file provides the classes for 2D contour plotting.

"""

#%% Modules:

import numpy as np
shape = np.shape
reshape = np.reshape
size = np.size

import matplotlib.pyplot as plt

from UtilityFunc import UF
uf = UF()


#%% Contour plot class:

class ContourPlot():
    """Class to plot the contours of a given function."""
    
    def __init__(self,
                 domain,
                 tInterval=None,
                 discNum=51):
        """
        Initializer for the contour plot.
        
        Inputs:
            domain: an insatnce of Domain class containing domain information.
            tInterval [1x2]: time interval (default: time-independent)
            discNum: number of spatial discretization points
        
        Attributes:
            status: status of the problem whose plots are requested:
                '1D-time': 1D time-dependent problem
                '2D': 2D time-independent problem
                '2D-time': 2D time-dependent problem
            isOutside: True for points that do not lie inside domain
            x_coord: 1D discretization of the x-ccordinate
            y_coord: 1D discretization of the y-ccordinate
            X_coord: x-ccordinate of the meshgrid stacked in a column
            Y_coord: y-ccordinate of the meshgrid stacked in a column
                (y-coordinate may refer to time or 2nd coordinate in 2D problems)
            xx: x-coordinate in meshgrid format
            yy: y-coordinate in meshgrid format
        """
        dim = domain.dim
        lim = domain.lim
        hx = (lim[1,0] - lim[0,0])/(discNum-1)                              # element size
        x_coord = np.linspace(lim[0,0], lim[1,0], discNum)                  # x-discretization
        
        if dim==1 and uf.isnone(tInterval):
            raise ValueError('contour plot unavailable for 1D, time-independent problems!')
        elif dim==1:
            status = '1D-time'
            hy = (tInterval[1] - tInterval[0])/(discNum-1)                  # element size
            y_coord = np.linspace(tInterval[0], tInterval[1], discNum)      # t-discretization
        
        if dim==2:
            hy = (lim[1,1] - lim[0,1])/(discNum-1)                          # element size
            y_coord = np.linspace(lim[0,1], lim[1,1], discNum)              # y-discretization
            if uf.isnone(tInterval):
                status = '2D'
            else:
                status = '2D-time'
        
        # Mesh grid:
        xx, yy = np.meshgrid(x_coord, y_coord, sparse=False)
        
        # Function input:
        X_coord = np.tile(x_coord, discNum)                                 # copy for y
        X_coord = reshape(X_coord, [len(X_coord), 1])
        Y_coord = np.repeat(y_coord, discNum)                               # copy for x
        Y_coord = reshape(Y_coord, [len(Y_coord), 1])
             
        # Determine the points that lie outside the domain:
        if status=='1D-time':
            isOutside = np.zeros(discNum**2, dtype=bool)
        else:
            Input = np.concatenate([X_coord, Y_coord], axis=1)
            isOutside = np.logical_not(domain.isInside(Input))
        
        # Store data:
        self.status = status
        self.discNum = discNum
        self.tInterval = tInterval
        self.he = np.array([hx, hy])
        self.isOutside = isOutside
        self.x_coord = reshape(x_coord, [discNum,1])
        self.y_coord = reshape(y_coord, [discNum,1])
        self.X_coord = X_coord
        self.Y_coord = Y_coord
        self.xx = xx
        self.yy = yy
        self.domain = domain
        
        
    def conPlot(self, func, t=None, figNum=None, title=None, fill_val=0.):
        """
        Function to plot the contour field.
        
        Inputs:
            func: callable function of (x,t)
            t: time instance for 2D time-dependent problems
            title [string]: contour plot title
            figNum: figure number to draw on
            fill_val: value to be used for obstacles
            
        Note that the function 'func' must handle the obstcles by assigning 
        neutral values to the grid over the obstacles.
        """
        if not callable(func):
            raise ValueError('field function must be callable!')
        if self.status=='2D-time' and uf.isnone(t):
            raise ValueError('time must be provided for 2D time-dependent problems!')
            
        status = self.status
        discNum = self.discNum
        isOutside = self.isOutside
        X_coord = self.X_coord
        Y_coord = self.Y_coord
        domain = self.domain
        
        # Construct the field:
        if status=='1D-time':
            field = func(X_coord, Y_coord)
        elif status=='2D':
            Input = np.concatenate([X_coord, Y_coord], axis=1)
            field = func(Input)
        elif status=='2D-time':
            Input = np.concatenate([X_coord, Y_coord], axis=1)
            field = func(Input,t)
            
        # Process the field:
        if not shape(field)[0]==discNum**2:
            raise ValueError('output of the function should be a column vector with size {}!'.format(discNum**2))
        elif size(shape(field))==1:
            field = reshape(field, [discNum**2,1])
        field[isOutside,:] = fill_val
        field = np.reshape(field, [discNum, discNum])
        
        # Create the figure and plot the domain frame:
        if uf.isnone(figNum):
            figNum=0
        plt.figure(figNum)
        if domain.dim>1:
            domain.domPlot(addDescription=False, figNum=figNum, frameColor='w')
        
        # Plot the contour field:
        cP = plt.contourf(self.xx, self.yy, field)
        plt.colorbar(cP)
        if status=='1D-time':
            plt.xlabel('$x$')
            plt.ylabel('time')
        else:
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
        if not uf.isnone(title):
            plt.title(title)
        plt.axis('scaled')
        
        return field
        
        
    def animPlot(self, func, t=[], figNum=None, title=None, fill_val=0.):
        """
        Function to plot the animation of 2D time-dependent field.
        
        Inputs:
            func: callable function of (x,t)
            t: time instance vector for 2D time-dependent problems
        """
        # Error handling:
        if not callable(func):
            raise ValueError('field function must be callable!')
        if self.status=='1D-time' or self.status=='2D':
            raise ValueError('animation contour plot is only available for 2D time-dependent problems!')
        if uf.isnone(figNum):
            figNum=0

        # Data:
        discNum = self.discNum
        X_coord = self.X_coord
        Y_coord = self.Y_coord
        isOutside = self.isOutside
        domain = self.domain
        
        Input = np.concatenate([X_coord, Y_coord], axis=1)
        
        # If time sequence is not provided:
        if np.size(t)==0:
            tInterval = self.tInterval
            t = np.linspace(tInterval[0], tInterval[1], num=5)
                
        # Loop over time:
        for ti in t:
            plt.figure(figNum)
            domain.domPlot(addDescription=False, figNum=figNum, frameColor='w')
            
            field = func(Input, ti)
            
            # Process the field:
            if not shape(field)[0]==discNum**2:
                raise ValueError('output of the function should be a column vector with size {}!'.format(discNum**2))
            elif size(shape(field))==1:
                field = reshape(field, [discNum**2,1])
            field[isOutside,:] = fill_val
            field = np.reshape(field, [discNum, discNum])
            
            # Contour plot:
            cP = plt.contourf(self.xx, self.yy, field)
            plt.colorbar(cP)
            
            titleT = 't = {0:.2f}s'.format(ti)
            if not uf.isnone(title):
                title2 = title + '-' + titleT
            else:
                title2 = titleT
            plt.title(title2)
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.axis('scaled')
            plt.show()
            plt.pause(1)        # pause 1sec before plotting the next contour

        
    def snap1Dt(self, func, t, lineOpt=None, figNum=None, title=None):
        """
        Function to plot snapshots for 1D time-dependent function.
        
        Inputs:
            func: callable function of (x,t)
            t: vector of time instances corresponding to the snapshot
            lineOpt: line options to allow comparison between different functions
            figNum: figure number to draw on
        """
        
        # Error handling:
        if not callable(func):
            raise ValueError('field function must be callable!')
        if not self.status=='1D-time':
            raise ValueError('Function is specific to 1D time-dependent problems!')
            
        x_coord = self.x_coord
        field = func(x_coord, t)
        
        if uf.isnone(figNum):
            plt.figure()
        else:
            plt.figure(figNum)
        
        if uf.isnone(lineOpt):
            plt.plot(x_coord, field)
        else:
            plt.plot(x_coord, field, lineOpt)
            
        plt.xlabel('$x$')
        if not uf.isnone(title):
            plt.title(title)
        plt.grid(True)
        
        
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        