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
This file provides contains the model order reduction (MOR) class.

"""

#%% Modules:

import numpy as np
import numpy.linalg as la
shape = np.shape
reshape = np.reshape
size = np.size

from UtilityFunc import UF
uf = UF()

import matplotlib.pyplot as plt

#%% Model Order Reduction Class:

class MOR():
    """Class to manage the Model Order Reduction (MOR) variables."""
    
    def __init__(self,
                 funcHandles,
                 ArgNames,
                 ArgRange
                 ):
        """
        Function to initialize the attributes of the class.
        
        Inputs:
            funcHandles [list]: name of the functions in the input data whose 
                parameters are variable (NN will be trained to solve the problem
                for the whole range of these variables)
            ArgNames [list of lists]: correspnding argument names of each 
                function included
            ArgRange [list of matrices]: list of matrices corresponding to the 
                range of arguments
                
        Attributes:
            
        """
        # Error handling:
        if not type(funcHandles)==list and callable(funcHandles):
            funcHandles = [funcHandles]
            ArgNames = [ArgNames]
            ArgRange = [ArgRange]
        elif not type(funcHandles)==list:
            raise ValueError('\'funcHandles\' must be a list of callable functions!')
            
        # Initialization:
        funNum = len(funcHandles)                               # number of function handles
        varNum = []                                             # number of variable arguments in each function
        argInd = []                                             # index of each argument among the relevant function's arguments
        sortInd = []                                            # sort indices for arguments of each function
        
        for i in range(funNum):                                 # loop over functions
            func = funcHandles[i]
            if not callable(func):
                raise ValueError('entries must be callable functions!')
                
            # Extract function info:
            funName = func.__code__.co_name                     # name of the function
            argNum = func.__code__.co_argcount                  # number of arguments of the function
            varNames = func.__code__.co_varnames[0:argNum]      # names of the variables

            # Convert single arguments to lists:
            if type(ArgNames[i]) is not list:
                ArgNames[i] = [ArgNames[i]]
                ArgRange[i] = [ArgRange[i]]
            
            # Check that the name of the arguments are valid:
            argNames = ArgNames[i]                              # names of variable arguments
            varNum0 = len(argNames)
            argInd0 = []
            for j in range(varNum0):                            # loop over variable arguments of current function
                try:
                    argInd0.append(varNames.index(argNames[j])) # find the index of argument
                except:
                    raise ValueError(argNames[j] + ' is not an argument of ' + funName + '!')
                
            # Variable arguments must be the last ones in the argument list of the function:
            ind = np.argsort(argInd0)
            argInd0 = uf.reorderList(argInd0, ind)
            ArgNames[i] = uf.reorderList(ArgNames[i], ind)
            if not (argInd0[-1]==argNum-1 and len(argInd0)==argInd0[-1]-argInd0[0]+1):
                raise ValueError('variable arguments of ' + funName + 
                                 ' must be ordered and the last arguments to the function')
                
            # Update the argument information for current function:
            varNum.append(varNum0)
            argInd.append(argInd0)
            sortInd.append(ind)
            
            # Check the dimension agreement of the ranges:
            if shape(ArgRange[i])[1]!=2:
                raise ValueError('dimension of the variable ranges for function ' + funName + 
                                 'are not equal to 2!')
            elif len(ArgRange[i])!=varNum0:
                raise ValueError('number of variable ranges for function ' + funName + 
                                 'does not match the number of variable arguments!')
            
            ArgRange[i] = uf.reorderList(ArgRange[i], ind)      # reorder argument range
            
        # Store attributes:
        self.funNum = funNum
        self.funcHandles = funcHandles
        self.ArgNames = ArgNames
        self.ArgRange = ArgRange
        self.varNum = varNum
        self.argInd = argInd
        self.sortInd = sortInd
        
        
    def discretizeArg(self, discScheme, randFlag=False):
        """
        Function to discretize the variable arguments of the functions.
        
        Input:
            discScheme [list]: list of argument discretization schemes for each
                function, each item in the list could be:
                    - a scalar specifying the same number of discretizations for
                        all variable arguments
                    - a column vector specifying the number of discretizations per
                        argument (used to select training points)
                    - a function handle to discretize the arguments whose values
                        are defined in relation with each other (they are dependent),
                        e.g., for the source support the upper bound in each 
                        dimension is strictly larger than the lower bound.
                        The function returns a matrix of all combinations of the
                        discretized values for all arguments stored in columns
            randFlag [bool]: if true use uniform random sampling instead of 
                uniform discretization of the variable arguments
                
        Output:
            discArg [list]: each entry of the list is a list of column vectors
                corresponding to the discretization of one of the variable 
                arguments
        """
        funcHandles = self.funcHandles          # list of function handles
        funNum = self.funNum                    # number of function handles
        varNum = self.varNum                    # number of variable arguments
        ArgRange = self.ArgRange                # range of variable arguments             
        sortInd = self.sortInd                  # maping of arguments from user-specified to class-ordered
        
        # Error handling:
        if not type(discScheme)==list and callable(discScheme):
            discScheme = [discScheme]
        elif not type(discScheme)==list:
            raise ValueError('\'discScheme\' must be a list!')
        
        # Error handling: check the dimension agreement of the discNum:
        for i in range(funNum):                 # loop over functions
            func = funcHandles[i]
            funName = func.__code__.co_name     # name of the function
            if callable(discScheme[i]):
                continue
            elif size(discScheme[i])!=1 and size(discScheme[i])!=varNum[i]:
                raise ValueError('number of discretization numbers for function ' + funName + 
                                 ' does not match the number of variable arguments!')
            elif size(discScheme[i])==1:
                discScheme[i] = np.tile(discScheme[i], varNum[i])
        
        discArg = []                            # list containing discretized variable arguments
        for i in range(funNum):                 # loop over functions
            func = funcHandles[i]
            funName = func.__code__.co_name     # name of the function
            
            if callable(discScheme[i]):         # discretize via provided function
                discArg0 = discScheme[i]()      # call the function for discretization
                
                if not shape(discArg0)[1]==varNum[i]:
                    raise ValueError('output dimension of the function handle ' +
                                     'to discretize ' + funName + ' is not equal ' +
                                     'to its number of variable arguments!')
                discArg.append(discArg0)
                continue                        # continue to next function
            
            # Reorder the ranges according to the class-ordered indexing of arguments:
            ind = sortInd[i]
            discScheme[i] = uf.reorderList(discScheme[i], ind)
            
            # Discretize each argument and construct the combinations:
            discArg0 = []
            for j in range(varNum[i]):          # loop over variable arguments
                l0 = ArgRange[i][j][0]          # lower-bound
                l1 = ArgRange[i][j][1]          # upper-bound
                dof = discScheme[i][j]          # number of discretizations
                
                if randFlag:                    # discretize via random sampling
                    disc = np.random.uniform(l0, l1, dof)
                    disc = np.sort(disc)
                else:                           # discretize via uniform grid
                    disc = np.linspace(l0, l1, dof)
                disc = reshape(disc, [dof, 1])
                discArg0 = uf.pairMats(discArg0, disc)
            
            # Store discretized argument values for current function:
            discArg.append(discArg0)
            
        return discArg
    
    
    def argIndex(self, discArg):
        """
        Function to generate indices for the arguments of the functions provided
        to the MOR class. These indices specify the number of argument sample
        that is used for each function allowing a single for loop to handle the
        MOR training.
        
        Input:
            discArg [list]: discretized arguments for all functions
        """
        funNum = self.funNum                    # number of function handles        
        
        argInd = []                             # list containing discretized variable arguments
        for i in range(funNum):                 # loop over functions
            dof = len(discArg[i])               # number of discretizations
            ind = reshape(range(dof), [dof,1])
            argInd = uf.pairMats(argInd, ind)
            
        return argInd



    def POD(self, sqrtR=None, cShots=None, energy=None, K=None, pltEig=True, filepath=None):
        """
        Function to generate optimal POD basis for a given set of snapshots of
        the solution.
        
        Inputs:
            sqrtR: square root of the FE mass matrix for numerical integration
            cShots: snapshots of the AD-PDE for various realization of MOR arguments
            K: FE coefficient matrix
            energy: energy level kept for selection of basis number
            pltEig: plot the eigenvalues of the covariance matrix
            filepath: path to store and load POD basis functions from
        """
        # Error handling:
        if uf.isnone(cShots) and uf.isnone(filepath):
            raise ValueError('snapshots or a file path must be given to get the POD basis functions from!')
        if not uf.isnone(cShots) and uf.isnone(sqrtR):
            raise ValueError('\'sqrtR\' must be provided for MOR!')
            
        # Construct the POD basis functions:
        if not uf.isnone(cShots):
            kapa = shape(cShots)[0]                         # total number of snapshots
            Cmat = np.matmul(sqrtR,cShots)                  # sqrt of covariance matrix
            Cmat = Cmat/np.sqrt(kapa)
            _, S, Vh = la.svd(Cmat, full_matrices=False)    # compute only kapa left singular vectors
            D = S**2                                        # eigenvalues of covariance matrix
            PhiTot = np.matmul(cShots,Vh.T)
        else:
            D, PhiTot = np.load(filepath)                   # Load data
        
        # Save the POD basis function array:
        if not uf.isnone(filepath):
            np.save(filepath, (D, PhiTot))
        
        # Construct the reduced coefficient matrix:
        if not uf.isnone(K):
            KpTot = np.matmul(np.matmul(PhiTot.T,K.todense()), PhiTot)
        else:
            KpTot = None
        
        # Find the number of basis functions:
        if not uf.isnone(energy):
            sumD = np.cumsum(D)
            ind = sumD/sumD[-1] < energy
            basisNum = sum(ind)+1
            
            Phi = PhiTot[:, :basisNum]
            if not uf.isnone(K): Kp = KpTot[:basisNum, :basisNum]
        else:
            basisNum = None
            Kp = None
            
        # Plot the eigenvalues:
        if pltEig:
            plt.semilogy(D)
            plt.grid(True)
            plt.axvline(basisNum, color='r', linestyle='--')
            plt.xlabel('basis function index')
            plt.ylabel('basis function energy')
            plt.title('eigenvalue spectrum')
            plt.show()
        
        return Phi, Kp, basisNum, PhiTot, KpTot
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        