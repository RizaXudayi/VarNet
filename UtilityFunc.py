# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:11:51 2018

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
This file contains general utility functions used in the VarNet class.

"""

#%% Modules:

import os
import shutil
import numbers
import warnings

import math

import numpy as np
import numpy.linalg as la
pi = np.pi
sin = np.sin
cos = np.cos
shape = np.shape
size = np.size
reshape = np.reshape

import scipy as sp
from scipy import interpolate
import scipy.sparse as sparse
from scipy.linalg import block_diag

import csv

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

#%% Utility Function Class:

class UF():
    """Class to define custom functions to facilitate using Python."""

    def isnumber(self, x):
        """Checks if x contains all numbers."""
        if type(x)==list:
            x = self.unpackList(x)
            for x1 in x:
                if type(x1)==np.ndarray:
                    if not self.arrayIsnumber(x1): return False
                elif not isinstance(x1, numbers.Number):
                    return False
            return True
        
        elif type(x)==np.ndarray:
            return self.arrayIsnumber(x)
        
        else:
            return isinstance(x, numbers.Number)
    
    
    def arrayIsnumber(self, x):
        """Checks if an array contains all numbers."""
        if not type(x)==np.ndarray:
            raise ValueError('\'x\' must be a numpy array!')
            
        x = reshape(x, size(x))
        for x1 in x:
            if not isinstance(x1, numbers.Number):
                return False
        return True



    def isempty(self, x):
        """Checks if x is empty."""
        if type(x)==list and len(x)>0:
            return False
        elif type(x)==dict and len(x)>0:
            return False
        elif not (type(x)==list or type(x)==dict) and size(x)>0:
            return False
        else:
            return True
        
        
    def isnone(self, x):
        """Checks if x is none. Returns true even if a single element is None."""
        
        if self.isempty(x):
            return False
        
        elif type(x)==list:
            x = self.unpackList(x)
            for i in range(len(x)):
                if type(x[i]).__module__==np.__name__ and size(x[i])>1 and (x[i]==None).any():
                    return True
                elif type(x[i]).__module__==np.__name__ and size(x[i])==1 and x[i]==None:
                    return True
                elif type(x[i]).__module__==np.__name__:
                    continue
                elif x[i]==None:
                    return True
            return False
        
        elif type(x).__module__==np.__name__ and size(x)>1:
            return (x==None).any()
        
        else:
            return x==None
        
        
    def unpackList(self, x):
        """
        Retruns elements of a list with arbitraty structure, i.e., a list of 
        lists, in a single list.
        """
        # Handle error:
        if type(x)!=list:
            raise ValueError('Input argument must be a list!')
            
        outList = []
        for i in range(len(x)):
            if type(x[i])==list:
                outList.extend(self.unpackList(x[i]))
            elif not self.isempty(x[i]):
                outList.append(x[i])
        
        return outList
    
    
    def vstack(self, tup):
        """
        Function to stack tensors vertically. The numpy version does not accept
        any empty lists (arrays).
        """
        if self.isempty(self.unpackList(tup)): return []
        
        tup2 = []
        for i in range(len(tup)):
            if not self.isempty(tup[i]):
                tup2.append(tup[i])
                
        return np.vstack(tup2)
    
    
    def hstack(self, tup):
        """
        Function to stack tensors horizontally. The numpy version does not accept
        any empty lists (arrays).
        """
        if self.isempty(self.unpackList(tup)): return []
        
        tup2 = []
        for i in range(len(tup)):
            if not self.isempty(tup[i]):
                tup2.append(tup[i])
        return np.hstack(tup2)


    def nodeNum(self, x, val):
        """
        Function to find closest value in the vector 'x' to value 'val'.
        
        Inputs:
            x [n x dim]: column matrix of values
            val [m x dim]: vector of values to be found in 'x'
        """
        # Handle errors:
        if not type(x) in [list, np.ndarray]:
            raise TypeError('\'x\' must be a column matrix!')
        if type(x) is list: x = np.array(x)
        if not size(x.shape)==2: raise ValueError('\'x\' must be a column matrix!')
        dim = x.shape[1]
        
        if not self.isnumber(val):
            raise TypeError('entries of val must be a numbers!')
        if isinstance(val, numbers.Number): val = np.array([[val]])
        elif type(val) is list: val = np.array(val)
        
        vshape = val.shape
        if dim==1 and size(vshape)==2 and not vshape[1]==1:
            raise ValueError('dimension inconsistant!')
        elif dim==1: val=self.column(val)
        elif not size(vshape)==2 or not vshape[1]==dim: raise ValueError('dimension inconsistant!')
        
        ind = []
        for i in range(val.shape[0]):
            ind.append(np.argmin(la.norm(x-val[i, :], axis=1)))
        
        return ind


    def nodeNumGrid(self, obj, x=None, y=None, t=None):
        """
        Function to find the indices of a given point in a uniform grid over space-time.
        The spatial domain is assumed to be 2d.
        
        Inputs:
            obj: class instance with domain and discretization attributes:
                - spatial: domLims, discNum, dx, dy, domain
                - temporal: n, T, dt
            x, y, t: coordinates of the point in space-time
         
        Output:
            nx, ny, nt: indices of the point in the repsective multi-dimensional gird
            x, y, t: updated coordinates taking into account the discretization steps
            ns: spatial node number in a column 'coord' vector
        """
        # Data:
        dL = obj.domLims
        discNum = obj.discNum
        dx = obj.dx
        dy = obj.dy
        domain = obj.domain
        
        n = obj.n
        dt = obj.dt
        T = obj.T
        
        # Error handling:
        if not self.isnone(x) and self.Type(x) is not float: raise TypeError('\'x\' must be a float number!')
        if not self.isnone(y) and self.Type(y) is not float: raise TypeError('\'y\' must be a float number!')
        
        if not self.isnone([x,y]) and not domain.isInside(np.array([[x,y]]), tol=1e-6):
            raise ValueError('specified point is outside specified domain bounds!')
        if not self.isnone(x) and (x<dL[0,0] or dL[1,0]<x):
            raise ValueError('specified depth \'x\' is outside the domain bounds!')
        elif not self.isnone(y) and (y<dL[0,1] or dL[1,1]<y):
            raise ValueError('specified depth \'y\' is outside the domain bounds!')
            
        if not self.isnone(t) and self.Type(t) is not float: raise TypeError('\'t\' must be a float number!')
        elif not self.isnone(t) and (t<0.0 or t>T): raise ValueError('\'t\' is out of bound!')
                
        # Find closest discrete nodes to each coordinate:
        if not x is None:
            nx1 = math.floor((x - dL[0,0])/dx)
            nx2 = math.ceil((x - dL[0,0])/dx)
            x1 = dL[0,0] + nx1*dx
            x2 = dL[0,0] + nx2*dx
            if np.abs(x-x1)<=np.abs(x-x2): nx, x = nx1, x1
            else:                          nx, x = nx2, x2
            if nx==discNum[0]: nx-=1
        else:
            nx, x = [None]*2
            
        if not y is None:
            ny1 = math.floor((y - dL[0,1])/dy)
            ny2 = math.ceil((y - dL[0,1])/dy)
            y1 = dL[0,1] + ny1*dy
            y2 = dL[0,1] + ny2*dy
            if np.abs(y-y1)<=np.abs(y-y2): ny, y = ny1, y1
            else:                          ny, y = ny2, y2
            if ny==discNum[1]: ny-=1
        else:
            ny, y = [None]*2
            
        if not t is None:
            nt1 = math.floor(t/dt)
            t1 = nt1*dt
            nt2 = math.ceil(t/dt)
            t2 = nt2*dt
            if np.abs(t-t1)<=np.abs(t-t2): nt, t = nt1, t1
            else:                          nt, t = nt2, t2
            if nt==n: nt-=1
        else:
            nt, t = [None]*2
        
        if not self.isnone([x,y]):  ns = ny*discNum[0]+nx
        else:                       ns = None
        
        return nx, ny, nt, x, y, t, ns


    def pairMats(self, mat1, mat2, reverse=False):
        """
        Utility function to pair matrices 'mat1' and 'mat2' by tiling 'mat2' and
        repeating rows of 'mat1' for each tile of 'mat1'.
        
        Inputs:
            mat1 [n1xm1]
            mat2 [n2xm2]
            reverse: if True pile 'mat1' and repeat the entries of 'mat2'
            
        Output:
            MAT [(n1*n2)x(m1+m2)]
        """
        # Error handling:
        if self.isempty(mat1):
            return mat2
        elif self.isempty(mat2):
            return mat1
        
        # Swap 'mat1' and 'mat2':
        if reverse: mat = mat1; mat1 = mat2; mat2 = mat
        
        # Matrix dimensions:
        sh1 = shape(mat1)
        sh2 = shape(mat2)
        
        # Repeat one row of the first matrix per tile of second matrix:
        ind = np.arange(0, sh1[0])[np.newaxis].T
        ind = np.tile(ind, reps=[1, sh2[0]])
        ind = reshape(ind, newshape=sh1[0]*sh2[0])
        MAT1 = mat1[ind]
        
        # Tile second matrix:
        MAT2 = np.tile(mat2, reps=[sh1[0],1])
        
        # Output:
        if not reverse: MAT = np.hstack([MAT1, MAT2])
        else:           MAT = np.hstack([MAT2, MAT1])
        return MAT
            

    def rejectionSampling(self, func, smpfun, dof, dofT=None):
        """
        Function to implement the rejection sampling algorithm to select 
        points according to a given loss function. If more than one loss function
        is used in 'func', dofT determines the number of nodes that belong to 
        each one of them.
        
        Inputs:
            func: function handle to determine the loss value at candidate points
            smpfun: function to draw samples from
            dof [mx1]: number of samples to be drawn for each segment
            dofT [mx1]: determines the segment length in the samples and function values
        """
        # Error handling:
        if isinstance(dof, numbers.Number) and not self.isnone(dofT):
            raise ValueError('\'dofT\' must be None for scalar \'dof\'')
        elif isinstance(dof, numbers.Number):
            dof = [dof]
        m = len(dof)
            
        if m>1 and self.isnone(dofT):
            raise ValueError('\'dofT\' must be provided when \'dof\' is a list!')
        
        # Rejection sampling procedure:
        maxfunc = lambda x,i: np.max(x)
        fmax = self.listSegment(func(), dofT, maxfunc)                  # maximum function values over the uniform grid
        def rejecSmp(val, i):
            """Function for rejection sampling to be assigned to listSegment()."""
            nt = len(val)                                               # number of samples
            
            # Uniform drawing for each sample to determine its rejection or acceptance:
            uniformVal = np.random.uniform(size=[nt,1])
            
            # Rejection sampling:
            ind = uniformVal < (val/fmax[i])                            # acceptance criterion
            return reshape(ind, nt)
            
        # Initialization:
        ns = [0 for i in range(m)]                                      # number of samples
        inpuT = [[] for i in range(m)]                                  # keep accepted samples
        flag = True
        while flag:
            # draw new samples:
            samples = smpfun()
            smpList = self.listSegment(samples, dofT)
            
            # Function value at randomly sampled points:
            val = func(samples)
            
            # Rejection sampling for each segment:
            ind = self.listSegment(val, dofT, rejecSmp)                 # accepted indices
            
            flag = False                                                # stopping criterion
            for i in range(m):
                inpuTmp = smpList[i][ind[i]]                            # keep accepted samples
                inpuT[i] = self.vstack([inpuT[i], inpuTmp])             # add to previously accepted samples
                ns[i] += np.sum(ind[i])                                 # update the number of optimal samples
                if not flag and ns[i]<dof[i]: flag=True
            
        for i in range(m):
            inpuT[i] = inpuT[i][:dof[i],:]                              # keep only 'dof' samples
        
        return np.vstack(inpuT)                                         # stack all samples together
        
        
    def listSegment(self, vec, segdof, func=None):
        """
        This function segemnts a vector of values into smaller pieces stored
        in a list and possibly apply 'func' to each segment separately.
        
        Inputs:
            vec [n x dim]: vector to be segmented
            segdof [mx1]: segmentation nodes (each entry specifies the NUMBER 
                   of nodes in one segment)
            func: function to be applied to segments separately - this function
                should accept a list and its index in the original list
        """
        n = len(vec)
        
        # Error handling:
        if self.isnone(segdof) and self.isnone(func):
            return [vec]
        elif self.isnone(segdof):
            return [func(vec,0)]
        elif isinstance(segdof, numbers.Number):
            segdof = [segdof]
            m = 1
        else:
            m = len(segdof)
            
        if segdof[-1]>n:
            raise ValueError('\'segdof\' is out of bound!')
            
        # Segmentation:
        outVec = [[] for i in range(m)]
        ind = 0
        for i in range(m):
            if not self.isnone(func):
                outVec[i] = func(vec[ind:(ind+segdof[i])][:], i)
            else:
                outVec[i] = vec[ind:(ind+segdof[i])][:]
            ind += segdof[i]
            
        # Add the remainder if it exists:
        if ind<n and not self.isnone(func):
            outVec.append( func(vec[ind:], i) )
        elif ind<n:
            outVec.append(vec[ind:])
        
        return outVec
        
        
    def reorderList(self, x, ind):
        """Reorder the entries of the list 'x' according to indices 'ind'."""
        n = len(x)
        
        # Error handling:
        if not len(ind)==n:
            warnings.warn('length of the indices is not equal to the length of the list!')
        if not self.isnumber(ind):
            raise ValueError('\'ind\' must be a list of integers!')
            
        if type(ind)==np.ndarray:
            ind = reshape(ind, n)
            
        return [x[i] for i in ind]
            
        
    def buildDict(self, keys, values):
        """Build a dict with 'keys' and 'values'."""
        n = len(keys)
        
        # Error handling:
        if not len(values)==n:
            raise ValueError('length of the keys and values must match!')
            
        mydict = {}
        for i in range(n):
            mydict[keys[i]] = values[i]
            
        return mydict
        
    
    def l2Err(self, xTrue, xApp):
        """Function to compute the normalized l2 error."""
        
        # Preprocessing:
        if sparse.issparse(xTrue): xTrue = xTrue.todense()
        if sparse.issparse(xApp): xApp = xApp.todense()
        n = size(xTrue)
        if size(shape(xTrue))==1:
            xTrue = reshape(xTrue, [n,1])
        if not size(xApp)==n:
            raise ValueError('\'xTrue\' and \'xApp\' must have the same shape!')
        elif size(shape(xApp))==1:
            xApp = reshape(xApp, [n,1])
        
        return la.norm(xTrue-xApp)/la.norm(xTrue)
        
        
    def clearFolder(self, folderpath):
        """Function to remove the content of the folder specified by 'folderpath'."""
        
        if self.isempty(os.listdir(folderpath)): return
        
        # Make sure that the call to this function was intended:
        while True:
            answer = input('clear the content of the folder? (y/n)\n')
            if answer.lower()=='y' or answer.lower()=='yes':
                break
            elif answer.lower()=='n' or answer.lower()=='no':
                return
        
        for file in os.listdir(folderpath):
            path = os.path.join(folderpath, file)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(e)
        
        
    def copyFile(self, filename, folderpath):
        """
        Function to backup the operator settings for later reference.
        Inputs:
            filename: name of the current operator file
            folderpath: the destination folder path
        """
        if not os.path.exists(filename):
            filename = os.path.join(os.getcwd(), filename)
            if not os.path.exists(filename):
                raise ValueError('The file does not exist!')
        
        shutil.copy2(filename, folderpath)      # copy the file


    def polyArea(self, x, y=None):
        """
        Function to compute the area of a polygon using Shoelace formula.
        
        Inputs:
            x: vector of first coordinates or all coordinates in columns
            y: vector of second coordinates
        """
        if self.isnone(y) and not shape(x)[1]==2:
            raise ValueError('input must be 2d!')
        elif self.isnone(y):
            y = x[:,0]
            x = x[:,1]
        elif len(shape(x))>1 and not shape(x)[1]==1:
            raise ValueError('\'x\' must be a 1d vector of first coordinates!')
        elif not len(x)==len(y):
            raise ValueError('\'x\' and \'y\' must be the same length!')
        else:
            x = reshape(x, len(x))
            y = reshape(y, len(x))
        
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        
        
    def mergeDict(self, dictList): 
        """
        Function to merge a list of dictionaries into one dictionary.
        
        Input: list of dictionaries
        """
        # Error handling:
        if not type(dictList)==list:
            raise ValueError('input must be a list of dictionaries!')
            
        if len(dictList)==1:
            return dictList[0]
        elif len(dictList)==2:
            return {**dictList[0], **dictList[1]}
        else:
            return {**dictList[0], **self.mergeDict(dictList[1:])}
            
        
    def csvRead(self, filename):
        """Function to read csv file into a list."""
        
        if not os.path.exists(filename):
            filename = os.path.join(os.getcwd(), filename)
            if not os.path.exists(filename):
                raise ValueError('The csv file does not exist!')
        
        with open(filename, 'r') as file:
          reader = csv.reader(file)
          return list(reader)
        
        
    def column(self, vec):
        """Takes in a vector and returns a column vector."""
        if type(vec) is list: vec = np.array(vec)
        sh1 = vec.shape
        n = vec.size
        if sh1==(n,1): return vec
        elif sh1==(1,n): return vec.T
        elif sh1==(n,): return vec.reshape([n,1])
        else: raise ValueError('\'vec\' must be a vector!')
        
        
    def blkdiag(self, tup, empty=True):
        """
        Constructs the block-diagonal matrix of matrices in tuple 'tup'.
        
        Inputs:
            tup: tuple of matrices to used for block-diagonalization
            empty: if True remove potential empty entries from 'tup'
        """
        if empty: tup = self.rmEmptyList(tup)
        if tup==[]:       return []
        elif len(tup)==1: return np.array(tup[0])
        else:
            return block_diag(tup[0], self.blkdiag(tup[1:], empty=False))
        
        
    def rmEmptyList(self, val):
        """Removes empty entries from given list."""
        if type(val) is not list: raise ValueError('\'val\' must be a list!')
        if val==[]: return []
        
        val1 = []
        for v in val:
            if not self.isempty(v): val1.append(v)
        return val1
        
    
    def Type(self, x):
        """
        Function to return variable types considering different types in Python
        that essentially refer to the same thing from partical point of view.
        """
        # Type: integer
        if type(x) in [int, np.int, np.int0, np.int8, np.int16, np.int32, np.int64]:
            return int
        
        # Type: float
        if type(x) in [float, np.float, np.float16, np.float32, np.float64]:
            return float
        
        # Type: complex
        if type(x) in [complex, np.complex, np.complex64, np.complex128]:
            return complex
        
        return type(x)
        
        
    def addNoise(self, sig, delta, distn='Gaussian', method='additive'):
        """
        Function to generate simulated noisy data given a signal.
        
        Inputs:
            sig [nx1]: column vector of signal values
            delta: noise variation:
                Gaussian: std of the distn
                uniform: range of the distn
            distn: distribution of the noise: Gaussian or uniform
            method:
                additive: add noise to components
                multiplicative: add noise proportional to signal magnitude
                
        Outputs:
            sig: noise-corrputed signal
            SNR: signal-to-noise ratio
            noise: added noise vector
        """
        # Error handling:
        if not self.isnumber(sig): raise TypeError('signal must contain only numbers!')
        sig = self.column(sig)
        if self.Type(sig[0,0]) is float: Float = True
        elif self.Type(sig[0,0]) is complex: Float = False
        else: raise TypeError('entries of \'sig\' must be float or complex!')
        
        if not self.isnumber(delta): raise TypeError('\'delta\' must be a number!')
        if self.Type(delta) is float and delta<0.0:
            raise ValueError('\'delta\' must be a positive number!')
        if Float and self.Type(delta) is list:
            raise TypeError('\'delta\' must be a number!')
        if self.Type(delta) is list and not len(delta)==2:
            raise ValueError('\'delta\' must have exactly two components!')
        if self.Type(delta) is list and (delta[0]<0.0 or delta[1]<0.0):
            raise ValueError('components of \'delta\' must be positive numbers!')
            
        if self.Type(distn) is not str: raise TypeError('\'distn\' must be a string!')
        if not distn.lower() in ['gaussian', 'uniform']: raise ValueError('unknown distribution!')
        
        if self.Type(method) is not str: raise TypeError('\'method\' must be a string!')
        if not method.lower() in ['additive', 'multiplicative']: raise ValueError('unknown method!')
        
        # Pre-processing:
        n = len(sig)
        if not Float and self.Type(delta) is float:
            delta = [delta]*2
            
        # Generate standard random noise:
        if distn.lower()=='gaussian':  noise = np.random.normal(size=[n,2])
        elif distn.lower()=='uniform': noise = np.random.uniform(-1.0, 1.0, size=[n,2])
            
        # Scale appropriately:
        if method.lower()=='additive':
            noise = delta*noise
            if Float: noise = noise[:,0:1]
            else:     noise = noise[:,0:1] + 1j*noise[:,1:2]
            
        elif method.lower()=='multiplicative':
            if Float: noise = sig*delta*noise[:,0:1]
            else:     noise = delta[0]*sig.real*noise[:,0:1] + 1.0j*delta[1]*sig.imag*noise[:,1:2]
        
        SNR = 20*np.log10(la.norm(sig)/la.norm(noise))
        sig2 = sig + noise
        
        return sig2, SNR, noise
        
        
    def stem3(self, x, y):
        """
        Function to stem plot in 3d.
        
        Inputs:
            x [nx2]: coordinates in 2d plane
            y [nx1]: corresponding values
        """
        # Error handling:
        if self.Type(x) is not np.ndarray: raise TypeError('\'x\' must be a column matrix!')
        if not size(x.shape)==2: TypeError('\'x\' must be a column matrix!')
        n, d = x.shape
        if not d==2: raise ValueError('\'x\' must have two columns!')
        
        if self.Type(y) is not np.ndarray: raise TypeError('\'y\' must be a column matrix!')
        y = y.reshape([-1])
        if not len(y)==n: raise ValueError('length of \'y\' does not match \'x\'!')
        
        # Figure:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        
        x1 = x[:,0]
        x2 = x[:,1]
        for x1i, x2i, yi in zip(x1, x2, y):
            line=art3d.Line3D(*zip((x1i, x2i, 0), (x1i, x2i, yi)), marker='o', markevery=(1, 1))
            ax.add_line(line)
            
        # Set the limits:
        delta = 0.2
        
        x1min = np.min(x1)
        x1max = np.max(x1)
        x1min -= delta*(x1max-x1min)
        x1max += delta*(x1max-x1min)
        
        x2min = np.min(x2)
        x2max = np.max(x2)
        x2min -= delta*(x2max-x2min)
        x2max += delta*(x2max-x2min)
        
        ymin = np.min(y)
        ymax = np.max(y)
        ymin -= delta*(ymax-ymin)
        ymax += delta*(ymax-ymin)
        
        print(x1min)
        print(x1max)
        print(x2min)
        print(x2max)
        print(ymin)
        print(ymax)
        
        ax.set_xlim3d(x1min, x1max)
        ax.set_ylim3d(x2min, x2max)
        ax.set_zlim3d(ymin, ymax)
        ax.view_init(elev=50., azim=35)
        
        
    def stepFun(self, cellVal, cellLims, points):
        """
        Function to select the cell that each point belongs to and assign the value
        of that cell to the point.
        
        Inputs:
            cellVal [nx1]: values at the centers of cells
            cellLims [nx2xd]: list of 2xd matrices containing the lower- and upper-bounds
                of the cells
            points [mxd]: points to be assigned to cells
            
        Output:
            val [mx1]: value of the step function at each point
        """
        
        # Error handling:
        if not self.Type(cellVal) in [list, np.ndarray]:
            raise TypeError('\'cellVal\' must be an array containing step values at the cells!')
        if self.Type(cellVal) is list: cellVal = np.array(cellVal)
        cellVal = cellVal.reshape([-1])
        n = len(cellVal)            # number of cells
        
        if not self.Type(cellLims) in [list, np.ndarray]:
            raise TypeError('\'cellLims\' must be an array containing the lower- and upper-bounds of the cells!')
        if self.Type(cellLims) is list: cellLims = self.vstack(cellLims)
        if not cellLims.shape[:2]==(n,2):
            raise ValueError('\'cellLims\' must be a %ix2xd dimensional matrix!' % n)
        dim = cellLims.shape[2]    # space dimension
        
        if not self.Type(points) in [list, np.ndarray]:
            raise TypeError('\'points\' must be an array containing the inquiry points!')
        if self.Type(points) is list: cellLims = self.vstack(points)
        if not cellLims.shape[1]==dim:
            raise ValueError('\'points\' must exactly have two columns!')
        m = len(points)             # number of inquiry points
        
        val = np.zeros([m,1])
        for i in range(n):          # loop over cells
            lim = cellLims[i]
            ind = np.ones(m)
            for d in range(dim):    # loop over dimensions
                ind *= (lim[0,d]<=points[:,d])*(points[:,d]<=lim[1,d])
            
            # Assign the value of current cell to points that are incide it:
            ind = np.array(ind, dtype=bool)
            val[ind,:] = cellVal[i]
            
        return val
            
            
    def mat2diffpack(self, filename, fieldname, mat):
        """
        Function to write 'mat' in MATLAB format of Diffpack as an m-file.
        The output file is then used by Diffpack to read the MATLAB matrix in.
        
        Inputs:
            filename: name of the output m-file
            fieldname: name of the field to be stored (used by Diffpack)
            mat: desired numpy array
        """
        # Error handling:
        if not self.Type(filename) is str:
            raise TypeError('\'filename\' must be a string!')
        elif '.m' not in filename:
            raise ValueError('\'filename\' must end with \'.m\'!')
        
        if not self.Type(fieldname) is str:
            raise TypeError('\'fieldname\' must be a string!')
        
        if not self.Type(mat) in [list, np.ndarray]:
            raise TypeError('\'mat\' must be an array!')
        elif self.Type(mat) is list: mat = np.array(mat)
        
        # Data:
        sh = shape(mat)                 # size of the data
        if size(sh)>2:
            raise ValueError('\'mat\' must be at most 2d!')
        elif size(sh)==2 and sh[1]==1:
            mat = mat.reshape([-1])     # remove second dimension
        
        if size(sh)==1 or sh[1]==1:
            # Header:
            string =  'datatype = \'real\';\n\n'
            string += 'length = ' + str(sh[0]) + ';\n\n'
            string += fieldname + ' = zeros(length,1);\n\n'
            string += '%% Data:\n\n'
            
            # Matrix entries:
            for i in range(sh[0]):      # loop over rows
                string += fieldname + '(' + str(i+1) + ') = \t' + str(mat[i]) + ';\n'
            
        else:
            # Header:
            string =  'datatype = \'real\';\n\n'
            string += 'nrows = ' + str(sh[0]) + '; ncolumns = ' + str(sh[1]) + ';\n'
            string += 'nentries = ' + str(np.prod(sh)) + ';\n\n'
            string += fieldname + ' = zeros(nrows, ncolumns);\n\n'
            string += '%% Data:\n\n'
            
            # Matrix entries:
            for j in range(sh[1]):      # loop over rows
                for i in range(sh[0]):  # loop over columns
                    string += fieldname + '(' + str(i+1) + ',' + str(j+1) + ') = \t' + str(mat[i,j]) + ';\n'
        
        # Write the file:
        with open(filename, 'w') as myfile:
            myfile.write(string)
            
            
    def meshNavigation(self, discNum, locind, step):
        """
        Function to navigate in a structured mesh. The function assumes that the
        numbering starts at the bottom-left corner of the domain and moves left
        to right and up toward the top-right corner.
        
        Inputs:
            discNum [1 x dim]: number of elements at each dimension
            locind: index location of the robot in the mesh
            step [1 x dim]: step vector
        """
        
        # Error handling:
        if not self.Type(discNum) in [list, np.ndarray]:
            raise TypeError('\'discNum\' must be a list!')
        if self.Type(discNum) is np.ndarray: discNum = discNum.reshape([-1])
        if not np.prod([self.Type(i)==int for i in discNum]):
            raise TypeError('\'discNum\' must only contain integers!')
        elif not (1<=len(discNum) or len(discNum)<=3):
            raise ValueError('\'discNum\' dimension incompatible!')
        
        # Data:
        dim = len(discNum)      # dimension of domain
        dof = np.prod(discNum)  # mesh noded number
        
        if not self.Type(locind) is int:
            raise TypeError('\'locind\' must be an inetger!')
        elif locind<0 or locind>=dof:
            raise TypeError('\'locind\' out of bound!')
            
        if not self.Type(step) in [list, np.ndarray]:
            raise TypeError('\'step\' must be a list!')
        if self.Type(step) is np.ndarray: step = step.reshape([-1])
        if not np.prod([self.Type(i)==int for i in step]):
            raise TypeError('\'step\' must only contain integers!')
        elif not len(discNum)==dim:
            raise ValueError('\'discNum\' dimension must be %!' % dim)
            
        if dim==3:
            warnings.warn('The code must be extended for dim=3!')
            return
        
        # 2D case:
        nextloc = locind + step[0]
        if step[0]>0 and (np.floor(locind/discNum[0])+1)*discNum[0]<=nextloc:   # '=' counts for indices starting from '0'
            raise ValueError('infeasible control step: robot location is outside domain!')
        elif step[0]<0 and np.floor(locind/discNum[0])*discNum[0]>nextloc:
            raise ValueError('infeasible control step: robot location is outside domain!')
                
        nextloc = nextloc + step[1]*discNum[0]
        if step[1]>0 and nextloc>=dof:                                          # '=' counts for indices starting from '0'
            raise ValueError('infeasible control step: next location is outside domain!')
        elif step[1]<0 and nextloc<0:
            raise ValueError('infeasible control step: next location is outside domain!')
                
        return nextloc
        
        
            
    def finiteDiff(self, dim, step, scheme='central'):
        """
        Function to construct a finite difference matrix that when multiplied with
        a column vector returns the finite difference values corresponding to each
        entry in a column vector with similar size.
        
        Inputs:
            dim: dimension of the discretized vector to be differentiated
            step: number of steps such that 'h = step*dx' where 'dx' is the mesh spacing
            scheme: finite difference scheme to be used:
                forward
                central
                backward
                
        Output [dim x dim]: desired sparse finite difference matrix
        """
        # Error handling:
        if not self.Type(dim) is int: raise TypeError('\'dim\' must be an integer!')
        elif dim<=0: raise ValueError('\'dim\' must be a positive integer!')
        
        if not self.Type(step) is int: raise TypeError('\'step\' must be an integer!')
        elif step<=0: raise ValueError('\'step\' must be a positive integer!')
            
        if not scheme in ['forward', 'central', 'backward']:
            raise ValueError('unrecognaized \'scheme\'!')
        if scheme in ['forward', 'backward']:
            print('these schemes are not developed in the code yet!')
            return
        
        if scheme=='central' and not np.mod(step,2)==0:
            raise ValueError('\'step\' must be an even number for central scheme!')
        
        n = (dim-step)*dim                  # number of enternal entries of the matrix
        D = np.zeros(n)
        ind = np.arange(0, n, dim+1, dtype=int)
        D[ind] = -1
        ind = np.arange(step, n, dim+1, dtype=int)
        D[ind] = 1
        D = D.reshape([dim-step, dim])      # enternal part of the matrix
        
        # Use forward and backeard scheme for the boundary nodes:
        step = int(step/2)
        D = np.vstack([D[:step,:], D, D[-step:,:]])
        
        return D
        
            
            
    def ANSYS(self, n, newcoord, filename, savepath=None):
        """
        Imports the data generated in ANSYS for 2D simulations and interpolates
        them over uniform spatial grids of given resolution and stores them in 
        the desired formats.
        
        Inputs:
            n: number of time steps
            newcoord: coordinates of the structured mesh used for interpolation
            filename: path and name of the text file to be read
            savepath: path to save the interpolated data - if not specified use
                the same folder as 'filenanme'
            
        Outputs: dictionary containing
            coord: coordinates of the original mesh
            time: time instances at which the data has been stored
            dispData: original displacement data
            newcoord: coordinates of the structured mesh used for interpolation
            newDispData: interpolated displacement data
            also stored in the same folder as filename with both numpy and MATLAB
            extensions
        """
        # Error handling:
        if not os.path.exists(filename):
            filename = os.path.join(os.getcwd(), filename)
            if not os.path.exists(filename):
                raise ValueError('The file does not exist!')
        if not filename.endswith('.txt'):
            raise TypeError('\'filename\' must specify a text file!')
        
        if self.isnone(savepath):
            k = filename.rfind('/')
            savepath = filename[:k]
        elif os.path.exists(filename):
            raise ValueError('\'savepath\' folder does not exist!')
            
        with open(filename, 'r') as file:
            data = file.read()
            
        # Read the mesh data:
        print('\nReading mesh data ...')
        k1 = data.find('\n')                # remove the header
        k2 = data.find('#')                 # find the first # indicator in the file
        coord = data[k1+1:k2-2].split()
        coord = np.fromiter(coord, float).reshape([-1,3])
        coord = coord[:,1:]                 # remove index numbers
        dof = coord.shape[0]                # number of spatial nodes
        print('Done! Number of nodes in ANSYS mesh: %i' % dof)
        
        # Read the displacement fields over time:
        k3 = k2 + data[k2:].find('\n')      # remove the header
        data = data[k3+1:]                  # remove mesh data
        dispData = np.zeros([dof,2,n])      # displacement fields over time
        newDispData = np.zeros([len(newcoord), 2, n])
        time = np.zeros(n)
        pval = 0.1                          # progress report
        print('\nReading displacement field data and interpolating ...')
        for ti in range(n):
            k1 = data.find('\n')            # remove the header
            k2 = data.find('#')             # find the first # indicator in the file
            dispTmp = data[k1+1:k2-1].split()
            dispTmp = np.fromiter(dispTmp, float).reshape([-1,4])
            time[ti] = dispTmp[0,1]         # current time
            dispTmp = dispTmp[:,2:]         # remove index numbers and time values
            dispData[:,:,ti] = dispTmp
            
            # Interpolate over new mesh:
            func = interpolate.CloughTocher2DInterpolator(coord, dispTmp, fill_value=0.0)
            newDispData[:,:,ti] = func(newcoord)
            
            # Report progress:
            if ti/n>=pval:
                print('%2.1f%% '% (pval*100.0), end='')
                pval += 0.1
            elif ti==n-1: print('100%\n')
            
            # Remove the read data:
            k3 = k2 + data[k2:].find('\n')  # remove the header
            data = data[k3+1:]              # remove mesh data
        
        # Check that no more data is left in the file to read:
        if not np.prod(val==' ' for val in data):
            raise ValueError('there is unread data in \'filename\', chek the value of \'n\'!')
        
        print('\nStoring the data ...')
        data = {'dof': dof,
                'coord': coord,
                'n': n,
                'time': time,
                'dispData': dispData,
                'newcoord': newcoord,
                'newDispData': newDispData}
        k1 = filename.rfind('/')
        k2 = filename.rfind('.')
        filename = filename[k+1:k2]
        savefile = os.path.join(savepath, filename + '.npy')
        np.save(savefile, data)             # numpy format
        savefile = os.path.join(savepath, filename + '.mat')
        sp.io.savemat(savefile, data)       # MATLAB format
        
        return data
            
            
    def inCircle(self, coord, cen, rad):
        """
        function to determine if a given set of points are in a circle.
        
        Inputs:
            coord [nx2]: coordinates of the points to be checked
            cen [1x2]: center of the circle
            rad: radius of the circle
        """
        # Error handling:
        if self.Type(coord) is not np.ndarray:
            raise TypeError('\'coord\' must be an array!')
        elif shape(coord)==(2,): coord = coord.reshape([1,-1])
        elif not shape(coord)[1]==2:
            raise ValueError('\'coord\' must be stored in an array with 2 columns!')
            
        if not self.Type(cen) in [list, np.ndarray]:
            raise TypeError('\'cen\' must be an array!')
        if self.Type(cen) is list: cen = np.array(cen)
        if len(shape(cen))==1: cen = reshape(cen, [1,2])
        elif not shape(cen)==(1,2): raise ValueError('\'cen\' must be stored in a row vector!')
        
        if not self.Type(rad) is float: raise TypeError('\'rad\' must be a float!')
        elif rad<=0: raise ValueError('\'rad\' must be positive!')
        
        # Compute the distance from the center:
        dist = la.norm(coord-cen, axis=1)
        return dist<=rad
    
    
    def inEllipse(self, coord, cen, a, b, theta):
        """
        Function to check if given points are inside the ellipse.
        
        Note: see https://en.wikipedia.org/wiki/Ellipse#General_ellipse
        
        Inputs:
            coord [nx2]: coordinates of the points to be checked
            cen: center of the circle
            a: horizontal semi-axis
            b: vertical semi-axis
            theta: angle (in degrees) of the horizontal axis from the x-axis
        """
        # Error handling:
        if self.Type(coord) is not np.ndarray:
            raise TypeError('\'coord\' must be an array!')
        elif shape(coord)==(2,): coord = coord.reshape([1,-1])
        elif not shape(coord)[1]==2:
            raise ValueError('\'coord\' must be stored in an array with 2 columns!')
            
        if not self.Type(cen) in [list, np.ndarray]:
            raise TypeError('\'cen\' must be an array!')
        if self.Type(cen) is list: cen = np.array(cen)
        if len(shape(cen))==2: cen = reshape(cen, 2)
        elif not shape(cen)==(2,): raise ValueError('\'cen\' must be stored in a row vector!')
        
        if not self.Type(a) is float: raise TypeError('\'a\' must be a float!')
        elif a<=0: raise ValueError('\'a\' must be positive!')
        
        if not self.Type(b) is float: raise TypeError('\'b\' must be a float!')
        elif b<=0: raise ValueError('\'b\' must be positive!')
        
        if not self.Type(theta) is float: raise TypeError('\'theta\' must be a float!')
        theta = theta*pi/180
        
        # General ellipse coefficients:
        A = a**2*sin(theta)**2 + b**2*cos(theta)**2
        B = 2*(b**2-a**2)*sin(theta)*cos(theta)
        C = a**2*cos(theta)**2 + b**2*sin(theta)**2
        D = -2*A*cen[0] - B*cen[1]
        E = -B*cen[0] - 2*C*cen[1]
        F = A*cen[0]**2 + B*cen[0]*cen[1] + C*cen[1]**2 - a**2*b**2
        
        # Compute the general ellipse equation at the desired coordinates:
        X = coord[:,0]
        Y = coord[:,1]
        val = A*X**2 + B*X*Y + C*Y**2 + D*X + E*Y + F
        return val<=0.
        
        
    def inTriangle(self, coord, corners):
        """
        function to determine if a given set of points are in a triangle.
        The method is based on barycentric coordinates; see
        https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
        
        Inputs:
            coord [nx2]: coordinates of the points to be checked
            corners [3x2]: coordinates of the corners of the triangle
        """
        # Error handling:
        if self.Type(coord) is not np.ndarray:
            raise TypeError('\'coord\' must be an array!')
        elif shape(coord)==(2,): coord = coord.reshape([1,-1])
        elif not shape(coord)[1]==2:
            raise ValueError('\'coord\' must be stored in an array with 2 columns!')
            
        if self.Type(corners) is not np.ndarray:
            raise TypeError('\'corners\' must be an array!')
        elif not shape(corners)==(3,2):
            raise ValueError('\'corners\' must be stored in an 3x2 array!')
        
        dx1 = corners[1,0] - corners[0,0]
        dx2 = corners[2,0] - corners[0,0]
        
        dy1 = corners[1,1] - corners[0,1]
        dy2 = corners[2,1] - corners[0,1]
        
        dX = coord[:,0:1]-corners[0,0]
        dY = coord[:,1:2]-corners[0,1]
        
        # Compute the s-t barycentric coordinates:
        A = np.array([[dx1, dx2], [dy1, dy2]])
        Ainv = la.inv(A)
        st = np.matmul(Ainv, np.vstack([dX.T, dY.T]))
        s = st[0,:]
        t = st[1,:]
        return (0<=s)*(s<=1)*(0<=t)*(t<=1)*(s+t<=1)
        
        
    def inRectangle(self, coord, corners):
        """
        function to determine if a given set of points are in a rectangle.
        The method is based on the method described in
        https://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle
        
        Inputs:
            coord [nx2]: coordinates of the points to be checked
            corners [4x2]: coordinates of the corners of the rectangle
        """
        # Error handling:
        if self.Type(coord) is not np.ndarray:
            raise TypeError('\'coord\' must be an array!')
        elif shape(coord)==(2,): coord = coord.reshape([1,-1])
        elif not shape(coord)[1]==2:
            raise ValueError('\'coord\' must be stored in an array with 2 columns!')
            
        if self.Type(corners) is not np.ndarray:
            raise TypeError('\'corners\' must be an array!')
        elif not shape(corners)==(4,2):
            raise ValueError('\'corners\' must be stored in an 4x2 array!')
        
        AM = coord - corners[0,:]
        ab = corners[1,:] - corners[0,:]
        ad = corners[3,:] - corners[0,:]
        
        abl = np.dot(ab, ab)
        adl = np.dot(ad, ad)
        AMab = np.dot(AM, ab)
        AMad = np.dot(AM, ad)
        
        return (0<=AMab)*(AMab<=abl)*(0<=AMad)*(AMad<=adl)
        
        
    def barPlot(self, data, ticks=None, labels=None, edgecolor='white', title=None):
        """
        Function to facilitate bar plots in a MATLAB fashion.
        
        Inputs:
            data [m x n]: 'n' sets of 'm' bar heights
            tickes [m x 1]: x-axis tick for each group
            labels [n x 1]: labels for each datum in each group
            edgecolor
            title
        """
        # Error handling:
        if not self.Type(data) in [list, np.ndarray]: raise TypeError('\'data\' must be a 2d array!')
        if self.Type(data) is list: data = np.array(data)
        if len(shape(data))==1: data = data.reshape([1,-1])
        m, n = shape(data)
        if m>7: raise ValueError('there are too many members in the groups!')
        
        if not self.isnone(ticks) and not len(ticks)==n:
            raise ValueError('the number of \'ticks\' is incompatible with the number of data points!')
        if self.isnone(ticks): ticks = range(1, n+1)
        
        if not self.isnone(title) and self.Type(title) is not str:
            raise TypeError('\'title\' must be a string!')
        
        # set width of bar
        barWidth = 0.25
         
        # x-axis:
        x = np.arange(n)
        if m>=4: x = 2*x
        xval = x
        
        # Loop over members of the groups:
        for j in range(m):
            if not self.isnone(labels): label = labels[j]
            else: label = None
            
            # Bar plot for member 'j' in all 'n' sets:
            plt.bar(xval, data[j,:], width=barWidth, edgecolor=edgecolor, label=label)
            
            # Uodate the x-axis for the next set of data:
            xval = [xi + barWidth for xi in xval]
        
        # Add the legend:
        if not self.isnone(labels): plt.legend()
        
        # Add xticks on the middle of the group bars:
        plt.xticks([xi + (m-1)/2*barWidth for xi in x], ticks)
         
        # Add title:
        if not self.isnone(title):
            plt.title(title)
        

    def checkBoundErr(self, interval, name, valType=float, valDefault=None, maxBound=None,
                 warn=True, selfErr=True):
        """
        Function to handle error in an interval assignment. It checks the given
        interval [a,b] for possible type and value errors.
        These family of functions whose name starts with 'err' are used as general
        error handlers for common input types.
        
        Inputs:
            interval: input argument by the user
            name: interval argument name to produce meaningful errors
            valType: data type of the entries
            valDefault: default value of the lower and upper-bound to be assigned 
                if one of the entries is given as 'None'
            maxBound: (minimum) lower and (maximum) upper-bound to be assigned 
                if one of the entries is out of bound
            warn: if True issue warnings when out of bound entries are set to default
            selfErr: if True check the arguments of this function for errors
        """
        # Error handling:
        if self.Type(selfErr) is not bool: raise TypeError('\'selfErr\' must be a boolean!')
        if selfErr:
            if self.Type(name) is not str: raise TypeError('\'name\' must be a string!')
            
            if not valDefault==None:
                if self.Type(valDefault) is not list: raise TypeError('\'valDefault\' must be a list!')
                elif not len(valDefault)==2: raise ValueError('\'valDefault\' must contain two entries!')
                elif not valDefault[0]==None and self.Type(valDefault[0]) is not valType:
                    raise TypeError('first entry of \'valDefault\' has inconsistent type!')
                elif not valDefault[1]==None and self.Type(valDefault[1]) is not valType:
                    raise TypeError('second entry of \'valDefault\' has inconsistent type!')
                elif not self.isnone(valDefault) and valDefault[1]<valDefault[0]:
                    raise ValueError('\'valDefault\' must contain two ordered numbers!')
                else: a, b = valDefault
            else: a, b = [None]*2
            
            if not self.isnone(maxBound):
                if self.Type(maxBound) is not list: raise TypeError('\'maxBound\' must be a list!')
                elif not len(maxBound)==2: raise ValueError('\'maxBound\' must contain two entries!')
                elif not maxBound[0]==None and self.Type(maxBound[0]) is not valType:
                    raise TypeError('first entry of \'maxBound\' has inconsistent type!')
                elif not maxBound[1]==None and self.Type(maxBound[1]) is not valType:
                    raise TypeError('second entry of \'maxBound\' has inconsistent type!')
                elif not self.isnone(maxBound) and maxBound[1]<maxBound[0]:
                    raise ValueError('\'maxBound\' must contain two ordered numbers!')
                else: l, u = maxBound
            else: l, u = [None]*2
            
            if (not self.isnone([l,a]) and a<l) or (not self.isnone([b,u]) and u<b):
                raise ValueError('\'valDefault\' is out of \'maxBound\'!')
            
            if self.Type(warn) is not bool: raise TypeError('\'warn\' must be a boolean!')
        
        else:
            if not valDefault==None: a, b = valDefault
            else: a, b = [None]*2
            
            if not self.isnone(maxBound): l, u = maxBound
            else: l, u = [None]*2
        
        # Interval error handling:
        if self.Type(interval) is not list:
            raise TypeError('\'' + name + '\' must be a list!')
        elif not len(interval)==2:
            raise ValueError('\'' + name + '\' must contain two entries!')
        
        if self.isnone(interval[0]):
            if a is not None: interval[0] = a
        else:
            if self.Type(interval[0]) is not valType:
                raise TypeError('lower bound for \'' + name + '\' must be a ' + valType.__name__ + '!')
            elif l is not None and interval[0]<l:
                interval[0] = l
                if warn: warnings.warn('lower bound for \'' + name + '\' is set to ' + str(l) + '!')
            
        if self.isnone(interval[1]):
            if b is not None: interval[1] = b
        else:
            if self.Type(interval[1]) is not valType:
                raise TypeError('upper bound for \'' + name + '\' must be a ' + valType.__name__ + '!')
            elif u is not None and u<interval[1]:
                interval[1] = u
                if warn: warnings.warn('upper bound for \'' + name + '\' is set to ' + str(u) + '!')
        
        if not self.isnone(interval) and interval[1]<interval[0]:
            raise ValueError('\'' + name + '\' must contain two ordered numbers!')
        
        return interval
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        