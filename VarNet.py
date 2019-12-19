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
This file contains the main class for variational Neural Networks that solve PDEs.

"""

#%% Modules:

import os
import math
import time
import warnings

import scipy.io as spio

import numpy as np
shape = np.shape
reshape = np.reshape
size = np.size

from scipy import interpolate

import matplotlib.pyplot as plt

import tensorflow as tf

from TFModel import TFNN
from VarNetUtility import RNNData, FIXData, ManageTrainData, TrainResult
from FiniteElement import FE
from ContourPlot import ContourPlot

from UtilityFunc import UF
uf = UF()

#%% Training class:
        
class VarNet():
    """Class to construct the relevant input and perform training for the NN model."""
    
    def __init__(self,
                 PDE,
                 layerWidth=[20],
                 modelId='MLP',
                 activationFun=None,
                 discNum=20,
                 bDiscNum=[],
                 tDiscNum=[],
                 MORdiscScheme=None,
                 processors=None,
                 controller=None,
                 integPnum=2,
                 optimizer='adam',
                 learning_rate=0.001):
        """
        Function to initialize the attributes of the class.
        
        Note that every time the VarNet class is instantiated, a new TensorFlow 
        graph is constructed and populated with computational nodes. This means  
        that repeating this process many times, slows down the code.
        
        Inputs:
            PDE: instance of ADPDE to be solved
            layerWidth [lNum x 1]: widths of the hidden layers
            modelId: indicator for the sequential TensorFlow model to be trained:
                'MLP': multi-layer perceptron with 'sigmoid' activation
                'RNN': recurrent network with 'gru' nodes
            activationFun: activation function used in the NN - options: 'sigmoid' or 'tanh'
            discNum [dim x 1]: list of training points at each dimension
            bDiscNum [bIndNum x 1]: list of training point density per unit length
            tDiscNum: time discretization number
            MORdiscScheme [list]: list of argument discretization schemes for each
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
            processors: processor(s) to be used for training (GPUs or CPU)
                        data is split between processors if more than one is specified
                        should be specified as 'CPU:i' or 'GPU:i' where i is 
                        the index of the processor
            controller (CPU or GPU): processor to contain the training data and
                        perform optimization in the parallel setting
            integPnum: number of integration points per dimension for each element
            optimizer: to be used for training: Adam, RMSprop
            learning_rate: learning rate for Adam optimizer
                
        Attributes:
            dim
            feDim: Finite Element integration dimension (one larger than 'dim'
                 for time dependent problems)
            fixData: fix data used throughout the implementation
            PDE
            layerWidth
            modelId
            discNum
            bDiscNum
            tDiscNum
            ht: time element size
            t_coord: time discretization
            graph: TensorFlow computational graph that stores the data
            inpDim: number of the NN inputs
            model: NN model
            processor: processor to be used for training (GPU or CPU)
        """
        # Data:
        dim = PDE.dim
        timeDependent = PDE.timeDependent
        MORvar = PDE.MORvar
        
        # Error handling:
        if size(discNum)!=1 and size(discNum)!=dim:
            raise ValueError('dimension of the number of discretizations does not' + 
                             ' match dimension of the domain!')
        elif size(discNum)==1:
            discNum = [discNum]*dim
        if size(bDiscNum)!=1:
            raise ValueError('density of boundary discretizations must be a scalar!')
            
        if not timeDependent and modelId=='RNN':
            raise ValueError('recurrent Neural net structure cannot used for ' +
                             'time-independent PDEs!')
        elif timeDependent and uf.isempty(tDiscNum):
            raise ValueError('time discretization number must be provided for ' +
                             'time-dependent PDEs!')
        
        if uf.isnone(activationFun):
            if modelId=='MLP':  activationFun = 'sigmoid'
            else:               activationFun = 'tanh'
            
        
        if type(layerWidth) is not list:
            raise ValueError('layer widths should be given in a list!')
        
        if not uf.isnone(MORvar) and uf.isnone(MORdiscScheme):
            raise ValueError('\'MORdiscScheme\' must be given for MOR!')
        
        # Determine the input dimension to the NN:
        inpDim = dim
        if PDE.timeDependent:
            inpDim += 1
        if not uf.isnone(MORvar):
            MORvarNum = MORvar.varNum
            for f in range(len(MORvarNum)):
                inpDim += MORvarNum[f]
        
        if integPnum==2: lossOpt = {'integWflag': False}
        else:            lossOpt = {'integWflag': True}
        if hasattr(PDE,'source') and PDE.source==0.0: lossOpt['isSource']=False
        else:                                         lossOpt['isSource']=True
        
        self.dim = dim
        self.discNum = discNum
        self.bDiscNum = bDiscNum
        self.tDiscNum = tDiscNum
        self.MORdiscScheme = MORdiscScheme
        self.modelId = modelId
        self.PDE = PDE
        if modelId=='MLP':
            self.fixData = FIXData(self, integPnum)             # store fixed data to save computation
            self.fixData.setInputData(self)                     # fixed PDE input data
            RNNdata = None
        elif modelId=='RNN':
            self.FixDataRNN(inpDim, integPnum)                  # generate the FE data for numerical integration
            RNNdata = self.RNNdata                              # RNN data including sequence length
        self.tfData = TFNN(dim, inpDim, layerWidth,
                           modelId, activationFun, timeDependent, RNNdata, 
                           processors, controller, lossOpt, optimizer, learning_rate)
        
        
        
    def FixDataRNN(self, inpDim, integPnum=2):
        """
        Function to generate the FE basis function data for RNNs.
        
        Input:
            integPnum: number of integration points in each dimension
        """
        
        # Data:
        dim = self.dim
        discNum = self.discNum
        bDiscNum = self.bDiscNum
        
        PDE = self.PDE
        timeDependent = PDE.timeDependent
        BCtype = PDE.BCtype
        domain = PDE.domain
        bIndNum = domain.bIndNum

        # Error handling:
        if not timeDependent:
            raise ValueError('the NN is not recurrent!')

        # Get the mesh for the time coordinate:
        tDiscNum = self.tDiscNum
        ht, seqLen, t_coord, tDiscInd, tIntegInd = self.timeDiscRNN(integPnum)
        
        # Get the mesh for the domain:
        mesh = domain.getMesh(discNum, bDiscNum)
        dof = mesh.dof                                          # number of training points inside domain
        bdof = mesh.bdof                                        # number of all boundary segment dofs
        he = mesh.he                                            # node sizes
        hVec = np.vstack([he, ht])                              # element sizes in space-time
        coord = mesh.coordinates                                # mesh coordinates for inner domain
        
        # Total boundary nodes (over space):
        bDof = 0
        for bInd in range(bIndNum):
            if BCtype[bInd] == 'Dirichlet':
                bDof += bdof[bInd]
        
        # Input to the NN for residual computations:
        # (smaller than training Input and uniform for easier interation):
        uniform_input = uf.pairMats(coord, t_coord)
        
        # Exact solution over the grid for error computation:
        if not uf.isnone(PDE.cEx):
            Coord = uniform_input[:,:dim]
            tCoord = uniform_input[:, dim:dim+1]
            cEx = PDE.cEx(Coord, tCoord)
        else:
            cEx = None
        
        # Fold into RNN input shape:
        uniform_input = reshape(uniform_input, [dof, seqLen, dim+1])
        
        # Uniform boundary input data:
        uniform_biInput, biDof = self.biTrainPoints(mesh, t_coord)
        
        # Total (Dirichlet) boundary discretization:
        bDofsum = np.sum(biDof[:-1])                            # exclude the initial condition dof
        
        # Processed FE basis data for RNNs:
        integNum, integW, delta, N, nablaPhi, deltaRNN = self.sortRNN(integPnum)
        nRNN = ((2*integPnum)**dim)*dof                         # number of spatial points for RNN input
        nt = dof*tDiscNum                                       # total number of training points in space-time
        nT = nt*integNum                                        # total number of integration points
        N = np.tile(N, reps=[nt, 1])                            # repeat basis values for trainig points
        detJ = np.prod(0.5*hVec)                                # Jacobian scaling of the integral
        
        # Basis derivatives in physical domain:
        dN = 2/hVec*nablaPhi                                    # derivative of the bases at integration points
        dN = np.tile(dN, reps=[1,nt]).T
        
        # Split spatial and temporal derivative values:
        dNx = dN[:,0:dim]
        dNt = dN[:,dim:dim+1]
        
        # Store data:
        self.fixData = FIXData(dim=dim, feDim=dim+1, integPnum=integPnum, integNum=integNum,
                             dof=dof, bdof=bdof, biDof=biDof, bDofsum=bDofsum, 
                             nt=nt, nT=nT, delta=delta, uniform_input=uniform_input,
                             uniform_biInput=uniform_biInput, cEx=cEx, N=N, dNx=dNx,
                             dNt=dNt, integW=integW, hVec=hVec, detJ=detJ)
        self.RNNdata = RNNData(dim, inpDim, integPnum, integNum, dof, bDof, tDiscNum,
                               nRNN, seqLen, nt, nT, t_coord, tDiscInd, 
                               tIntegInd, deltaRNN)

        
    
    def timeDisc(self, tdof=None, rfrac=0, sortflg=True, discTol=None):
        """
        Time discretization for time-dependent PDE.
        
        Inputs:
            tdof: number of time discretization points (use default if None)
            rfrac: fraction of samples that are drawn randomly
            sortflg: if True sort the randomly drawn samples
            discTol: minimum distance of discretization points from time lower bound
        
        Outputs:
            ht: time element size
            t_coord: time discretization
        """
        # Data:
        PDE = self.PDE
        
        # Error handling:
        if not PDE.timeDependent:
            raise Exception('The problem is time-independent!')
        
        # Project the random sampling fraction back to [0,1] interval:
        if rfrac<0:      rfrac = 0
        elif rfrac>1:    rfrac = 1
        
        if uf.isnone(tdof):
            tdof = self.tDiscNum
        tlim = PDE.tInterval
        
        ht = (tlim[1] - tlim[0])/tdof                               # time element size
        if uf.isnone(discTol):  tol = ht                            # discretization tolerance
        else:                   tol = np.asscalar(discTol)
        dof1 = math.floor(tdof*rfrac)                               # random grid
        dof2 = tdof - dof1                                          # uniform grid
        
        t_coord1 = np.random.uniform(tlim[0]+tol, tlim[1], dof1)    # random test function locations
        t_coord2 = np.linspace(tlim[0]+tol, tlim[1], dof2)          # uniform test function locations
        t_coord = uf.hstack([t_coord1, t_coord2])                
        if rfrac>0 and sortflg: t_coord = np.sort(t_coord)
        t_coord = reshape(t_coord, [tdof,1])
        
        return ht, t_coord
        
        
    
    def timeDiscRNN(self, integPnum):
        """
        The RNN includes fixed time-discretization with inquiry points for requested
        discrete times as well as numerical integration points. This function
        calculates the total sequence length for these inquiries which is also
        the length of the time dimension for the RNN and also constructs the 
        index sequence for extraction numerical integration data from the output
        of the RNN. It also gives the discrete time values corresponding to the 
        data that the RNN is trained on.
        
        Inputs:
            integP: number of integration points in time dimension
        
        Outputs:
            ht: time element size
            t_coord: time discretization
        """
        # Data:
        tDiscNum = self.tDiscNum
        
        # Error handling:
        if not self.modelId=='RNN':
            raise Exception('The NN is not recurrent!')
        
        # Construct the FE data for time dimension:
        fixData = FE(dim=1, integPnum=integPnum)
        delta = fixData.delta[0][0,:]
        
        # Add the inetrval end (discrete times) to query points if it is not part of the integration points:
        if np.sum(delta==0)==0:
            intDiscNum = integPnum+1
            delta = np.hstack([0, delta])
        else:
            intDiscNum = integPnum
        
        # Time discretization:
        ht, t_coord = self.timeDisc()
        t_coord = np.vstack([[0], t_coord])             # include t=0 into the discretized vector
        tDiscNum += 1
        
        # Total number of time discretizations (sequence length):
        seqLen = intDiscNum*tDiscNum
        
        # Time values for the RNN sequence:
        delta = np.tile(delta, reps=[tDiscNum,1])
        t_coord = t_coord + ht*delta
        t_coord = reshape(t_coord, newshape=[seqLen, 1])
        
        # Indices of discretization points:
        discInd = np.arange(intDiscNum, seqLen, intDiscNum)
        
        # Numerical integration over time involves copying the relevant nodes for spatial discretization points:
        integInd = []
        for i in range(1,integPnum+1):                  # loop over integration points
            start = intDiscNum+i                        # skip the first element
            stop = (tDiscNum-1)*intDiscNum              # skip the last element
            step = intDiscNum
            integInd.append( np.arange(start, stop, step) )
            
        integInd = np.array(integInd).T
        integInd = np.hstack([integInd, integInd])      # copy for each element
        integInd = reshape(integInd, newshape=[(tDiscNum-2)*integPnum*2, 1])
        
        # Include the first and last elements for numerical integration:
        ind = np.arange(1, integPnum+1)[np.newaxis].T
        integInd = np.vstack([ind, integInd])
        ind = np.arange(stop+1, stop+integPnum+1)[np.newaxis].T
        integInd = np.vstack([integInd, ind])
        integInd = reshape(integInd, newshape=len(integInd))
        
        return ht, seqLen, t_coord, discInd, integInd
        
        
    
    def sortRNN(self, integPnum):
        """Function to sort the FE basis data for RNNs."""
        
        # Data:
        dim = self.dim

        # FE basis data:        
        feDim = dim+1                                               # FE dimension for time-dependent problem
        feData = FE(feDim)                                          # construct the Finite Element bsais functions
        basisNum = feData.basisNum                                  # number of basis functions (equivalently elements)
        IntegPnum = feData.IntegPnum                                # number of integration points
        integNum = basisNum*IntegPnum                               # summation bound for numerical integration at each point
        integW = feData.integW                                      # numerical integration weights
        
        delta = reshape(feData.delta, [feDim, integNum])
        basVal = reshape(feData.basVal, newshape=[integNum, 1])     # basis values at integration points ()
        basDeriVal = reshape(feData.basDeriVal, newshape=[feDim, integNum])
        
        # Sort coordinates and update the corresponding basis values and derivatives:
        ind = np.lexsort( [ delta[i,:] for i in reversed(range(feDim)) ] )
        delta = delta[:,ind]
        basVal = basVal[ind]
        basDeriVal = basDeriVal[:,ind]
        
        # Extract the corresponding spatial translation vector:
        # (this only works if ordering in FE class is such that the last dimension is flipped first)
        ind = np.arange(0, integNum, step=IntegPnum, dtype=int)
        deltaRNN = delta[0:dim, ind]
        
        return integNum, integW, delta, basVal, basDeriVal, deltaRNN
        
        

    def trainingPointsRNN(self, mesh):
        """
        Function to construct the input to RNN corresponding to the training 
        data provided from the other functions in the class.
        """
        # Error handling:
        if not self.modelId=='RNN':
            raise ValueError('the NN is not recurrent!')
            
        # Data:
        dim = self.dim
        
        RNNdata = self.RNNdata
        nRNN = RNNdata.ns
        deltaRNN = RNNdata.deltaRNN
        
        PDE = self.PDE
        domain = PDE.domain
        BCtype = PDE.BCtype
        bIndNum = domain.bIndNum
        
        he = mesh.he
        coord = mesh.coordinates
        
        
        # Spatial integration points:
        coord2 = np.zeros([nRNN,dim])                           # store spatial coordinates of spatial integration points
        for d in range(dim):
            coordTmp = coord[:,d:d+1] + he[d]*deltaRNN[d,:]
            coord2[:,d] = reshape(coordTmp, nRNN)
            
        # Corresponding RNN input:
        Coord = RNNdata.buildInput(coord2)
        
        # Update the mesh struct to construct the initial coordinates over all points:
        mesh.dof = nRNN
        mesh.coordinates = coord2
        
        
        # Dirichlet boundary conditions:
        b_coord = mesh.bCoordinates                             # mesh coordinates for boundaries
        bInput = []
        for bInd in range(bIndNum):
            if BCtype[bInd] == 'Dirichlet':
                bInpuTmp = RNNdata.buildInput(b_coord[bInd])
                bInput.append(bInpuTmp)                         # append boundary input
        bInput = uf.vstack(bInput)
        
        # Put together the boundary and inner-domain inputs:
        InputRNN = uf.vstack([bInput, Coord])
        
        return InputRNN, mesh



    def trainingPoints(self, smpScheme='uniform', frac=0.5, addTrainPts=True, suppFactor=1.0):
        """
        Function to generate the training points and update the training points.
        
        Input:
            smpScheme: sampling scheme
                uniform: constant uniform samples
                random: randomly sample the space-time with a uniform distribution
                optimal: use feedback from PDE-residual to select optimal training points
            frac: fraction of training points selected optimally
            addTrainPts: if True add optimal training points when 'smpScheme=optimal',
                i.e., refine the mesh, o.w., keep the total number of training points constant
            suppFactor: support scaling for optimal training points if they are 
                added to already exisiting training points
        """
        # If optimal sampling scheme is requested, invoke a different function:
        if smpScheme=='optimal':
            return self.optTrainPoints(frac, addTrainPts, suppFactor)
        elif smpScheme=='random':
            rfrac = frac
        else:
            rfrac = 0.
        
        # Data:
        dim = self.dim
        discNum = self.discNum
        bDiscNum = self.bDiscNum
        modelId = self.modelId
        
        PDE = self.PDE
        timeDependent = PDE.timeDependent
        domain = PDE.domain
        
        fixData = self.fixData
        dof = fixData.dof
        nt = fixData.nt
        nT = fixData.nT
        delta = fixData.delta
        
        # Get the mesh for the time coordinate:
        if modelId=='RNN':
            ht = fixData.hVec[-1]
            RNNdata = self.RNNdata
            tDiscNum = RNNdata.tDiscNum
            tDiscInd = RNNdata.tDiscInd
            t_coord = RNNdata.t_coord[tDiscInd]                 # keep time-instances corresponding to numerical integration
        elif timeDependent:
            tDiscNum = self.tDiscNum
            ht, t_coord = self.timeDisc(rfrac=rfrac)
        else:
            tDiscNum = 1                                        # temporary value to create the integration points
            t_coord = []
        
        # Get the mesh for the domain:
        mesh = domain.getMesh(discNum, bDiscNum, rfrac=rfrac)
        if smpScheme=='random' and mesh.dof<dof:                # ensure that at the presence of obstacles exactly 'dof' samples are drawn
            dofTmp = mesh.dof
            coord = mesh.coordinates
            while dofTmp<dof:                                   # append random samples to reach dof discretization points
                meshTmp = domain.getMesh(discNum, bDiscNum, rfrac=1.)
                coord = uf.vstack([coord, meshTmp.coordinates])
                dofTmp += meshTmp.dof
                
            # Update the mesh attributes:
            mesh.dof = dof                                      # number of nodes in the inner domain
            mesh.coordinates = coord[:dof,:]                    # mesh coordinates for inner domain
            
        # Load mesh data:
        he = mesh.he                                            # node sizes
        coord = mesh.coordinates                                # mesh coordinates for inner domain
        
        # Integration points:
        Coord = np.zeros([nT,dim])                              # store spatial coordinates of integration points
        for d in range(dim):                                    # add integration points to spatial coordinates of each training point
            coordTmp = np.repeat(coord[:,d], repeats=tDiscNum)
            coordTmp = reshape(coordTmp, [nt, 1]) + he[d]*delta[d,:]
            Coord[:,d] = np.reshape(coordTmp, nT)               # re-arrange into a column
            
        if timeDependent:                                       # add integration points to time-coordinate of each node
            tCoord = np.tile(t_coord, reps=[dof,1])
            tCoord = tCoord + ht*delta[-1,:]
            tCoord = np.reshape(tCoord, [nT, 1])                # re-arrange into a column
            Input = np.concatenate([Coord, tCoord], axis=1)
        else:
            Input = Coord

        # Construct the input to RNN:
        if modelId=='RNN':
            t_coord = RNNdata.t_coord                           # for RNN use all time-discretization for BCs
            InputRNN, mesh = self.trainingPointsRNN(mesh)       # secondary mesh only used to generate initial condition
        else:
            InputRNN = []
        
        # Initial and boundary training data:
        biInput, biDof = self.biTrainPoints(mesh, t_coord)
        
        return Input, InputRNN, biInput, biDof
    
    

    def biTrainPoints(self, mesh, t_coord):
        """
        Function to prepare the training points for the boundary and initial
        conditions.
        
        Inputs:
            mesh: dictionary containing discretization information
                coord: coordinates of the discretization points in inner domain
                b_coord: list of boundary coordinates
            t_coord: time discretization
        """
        # Data:
        PDE = self.PDE
        timeDependent = PDE.timeDependent
        domain = PDE.domain
        BCtype = PDE.BCtype
        bIndNum = domain.bIndNum
        
        coord = mesh.coordinates                                        # mesh coordinates for inner domain
        dof = mesh.dof
        b_coord = mesh.bCoordinates                                     # mesh coordinates for boundaries
        
        # Initial condition:
        if timeDependent:
            iInput = np.concatenate([coord, np.zeros([dof,1])], axis=1)
        else:
            iInput = []
            
        # Dirichlet boundary conditions:
        bInput = []
        biDof = []
        for bInd in range(bIndNum):
            if BCtype[bInd] == 'Dirichlet':
                bInpuTmp = uf.pairMats(b_coord[bInd], t_coord)
                biDof.append(len(bInpuTmp))                             # number of boundary nodes
                bInput.append(bInpuTmp)                                 # append boundary input
        bInput = uf.vstack(bInput)
        
        if timeDependent: biDof.append(dof)                             # add nodes corresponding to initial condition
        
        biInput = uf.vstack([bInput, iInput])
        return biInput, biDof
    
    
    
    def biTrainData(self, biInput, biDof, biArg=[], biLabel0=[]):
        """
        Function to prepare the training data for the boundary and initial 
        conditions.
        
        Inputs:
            biInput: input value to the NN for boundary and initial condition
            biDof: length of each boundary-initial input
            biArg [list]: contains dictionaries for MOR variable arguments of 
                the boundary-initial condition functions
                Note: BICs will be computed without extra MOR arguments if biArg is empty
                      if each individual term in biArg is None, the corresponding BIC
                      will not be recomputed
            biLabel0: array containing the previous set of computed boundary-initial
                labels
        """
        # Data:
        dim = self.dim
        
        PDE = self.PDE
        timeDependent = PDE.timeDependent
        domain = PDE.domain
        BCtype = PDE.BCtype
        BCs = PDE.BCs
        bIndNum = domain.bIndNum
            
        # Default for MOR variable arguments:    
        if uf.isempty(biArg):
            biArg = [{} for i in range(bIndNum)]
            if timeDependent: biArg.append({})
        
        # Dirichlet boundary conditions:
        bLabel = []
        indp = 0
        for bInd in range(bIndNum):
            ind = indp + biDof[bInd]                    # current end index
            
            # Check if the computation of current boundary function is required:
            if not BCtype[bInd]=='Dirichlet':
                continue
            elif uf.isnone(biArg[bInd]) and uf.isempty(biLabel0):
                raise ValueError('\'biLabel0\' must be provided for \'None\' arguments!')
            elif uf.isnone(biArg[bInd]):
                bLabel.append(biLabel0[indp:ind,:])
                continue
            
            # Dirichlet BC data:
            beta = BCs[bInd][1]
            g = BCs[bInd][2]
            
            # Arguments:
            bCoord = biInput[indp:ind, :dim]
            if timeDependent:
                tCoord = [ biInput[indp:ind, dim][np.newaxis].T ]
            else:
                tCoord = []
                
            bLab = g(bCoord, *tCoord, **biArg[bInd])    # compute the boudary function
            bLabel.append(bLab/beta)                    # append corresponding label
            indp = ind                                  # update the previous index
        bLabel = uf.vstack(bLabel)
        
        # Initial condition:
        if timeDependent and uf.isnone(biArg[-1]) and uf.isempty(biLabel0):
            raise ValueError('\'biLabel0\' must be provided for \'None\' arguments!')
        elif timeDependent and uf.isnone(biArg[-1]):
            iLabel = biLabel0[ind:,:]
        elif timeDependent:
            coord = biInput[ind:, :dim]
            iLabel = PDE.IC(coord, **biArg[-1])
        else:
            iLabel = []
        
        return uf.vstack([bLabel, iLabel])
            


    def PDEinpData(self, Input, inpArg=[]):
        """
        Function to specify the parameterized PDE input data.
        
        Inputs:
            Input: coordinates of the numerical integration points in space-time
            inpArg [list]: containing the following dictionaries:
                diffArg [dict]: dictionary containing MOR variables for diffusivity function
                velArg [dict]: dictionary containing MOR variables for velocity function
                sourceArg [dict]: dictionary containing MOR variables for source function
                Note: input-data will be computed without extra MOR arguments if inpArg is empty
                      if each individual term in inpArg is None, the corresponding field will not
                      be recomputed
        """
        # Data:
        dim = self.dim
        PDE = self.PDE
        timeDependent = PDE.timeDependent
        diffFun = PDE.diffFun
        velFun = PDE.velFun
        sourceFun = PDE.sourceFun
        
        # Default for MOR variable arguments:
        if uf.isempty(inpArg):
            diffArg, velArg, sourceArg = [{} for i in range(3)]
        else:
            diffArg, velArg, sourceArg = inpArg
        
        if timeDependent:
            tCoord = [ Input[:,-1][np.newaxis].T ]
        else:
            tCoord = []
            
        if uf.isnone(diffArg):
            diff = None
        else:
            diff = diffFun(Input[:,0:dim], *tCoord, **diffArg)
        
        if uf.isnone(velArg):
            vel = None
        else:
            vel = velFun(Input[:,0:dim], *tCoord, **velArg)
            
        if uf.isnone(sourceArg):
            sourceVal = None
        else:
            sourceVal = sourceFun(Input[:,0:dim], *tCoord, **sourceArg)
        
        return diff, vel, sourceVal
    
    
    
    def trainData(self, batch, MORdiscArg, tData, resCalc=False):
        """
        Function to update the PDE input and boundary-inital data in a smart 
        way for computational savings. The function only updates segments of
        data if it has to change due to the update of space-time sampling and/or
        due to change in variable arguments of MOR.
        
        Inputs:
            tData: instance of 'ManageTrainData' class to store training data
            resCalc [bool]: determines if the data is being prepared for residual computations
        """
         
        # Default: return the data without update if space-time discretization
        #          has not changed and MOR variables do not exist:
        if not tData.inputUpdated and uf.isnone(MORdiscArg):
            return tData
        # Avoid recomputation if MOR data are stored:
        elif tData.MORdataSaved:
            tData.loadMORData(batch)
            return tData
        
        # Data:
        tfData = self.tfData
        Input, biInput, _, _, biLabel0, _, _, diff0, vel0 = tData.getAllData()
        
        fixData = self.fixData
        if not resCalc:
            nT = fixData.nT
        else:
            nT = shape(Input)[0]
        biDof = fixData.biDof
        N = fixData.N
        dNx = fixData.dNx
        
        
        # Update all data if the space-time discretization has been updated:
        if tData.inputUpdated:
            if not batch==0 and not resCalc:
                raise ValueError('\'batch\' variable must be reset when the ' + 
                                 'space-time discretization is updated!')
            elif uf.isnone(MORdiscArg):
                biArg, inpArg = [[]]*2
                MORinpNN = None
            else:
                biArg, inpArg, MORinpNN = self.MORargExtract(batch, MORdiscArg, defArg={})
                
            # Boundary-initial data:
            if not resCalc:
                biLabel = self.biTrainData(biInput, biDof, biArg)
            else:
                biLabel = []                                        # not required for residual calculations
                
            # PDE-input data:
            if resCalc and uf.isnone(MORdiscArg) and shape(fixData.uniform_input)[0]==nT:
                diff, vel, sourceVal = fixData.uniform_inpData
            else:
                diff, vel, sourceVal = self.PDEinpData(Input, inpArg)
                
            if not resCalc:
                gcoef = diff*dNx + vel*N
            else:
                gcoef = []                                          # not required for residual calculations
            
            # Construct the total input containing space-time as well as MOR discretizations:
            if not uf.isnone(MORdiscArg):
                InpMOR = np.tile(MORinpNN, reps=[nT,1])
                InpuTot = np.hstack([Input, InpMOR])
                if not resCalc:
                    InpMOR = np.tile(MORinpNN, reps=[np.sum(biDof),1])
                    biInpuTot = np.hstack([biInput, InpMOR])
                else:
                    biInpuTot = []                                  # not required for residual calculations
            else:
                InpuTot, biInpuTot = Input, biInput
            
            # Update the training data and return:
            tData.updateData(InpuTot, biInpuTot, biLabel, gcoef, sourceVal, diff, vel, MORinpNN)
            if not resCalc:
                tData.trainDicts(fixData, tfData)                  # generate training dictionaries
            return tData
        
        
        # Extract the relevant MOR arguments:
        biArg, inpArg, MORinpNN = self.MORargExtract(batch, MORdiscArg)
        
        # Update the boundary-initial condition data:
        funInd = self.PDE.MORfunInd                                 # mapping between PDE data and MOR functions
        if funInd['biData'] and not resCalc:
            biLabel = self.biTrainData(biInput, biDof, biArg, biLabel0)
        elif not resCalc:
            biLabel = None
        else:
            biLabel = []
        
        # Update the PDE input data:
        if funInd['inpData']:
            diff, vel, sourceVal = self.PDEinpData(Input, inpArg)
            
            if resCalc:           gcoef = []                        # not required for residual calculations
            elif uf.isnone(diff) and uf.isnone(vel):
                gcoef, diff, vel = [None]*3                         # none of the fields are updated
            elif uf.isnone(diff): gcoef = diff0*dNx + vel*N
            elif uf.isnone(vel):  gcoef = diff*dNx + vel0*N
            else:                 gcoef = diff*dNx + vel*N
            
        else:
            gcoef, sourceVal, diff, vel = [None]*4
        
        # Construct the total input containing space-time as well as MOR discretizations:
        InpMOR = np.tile(MORinpNN, reps=[nT,1])
        InpuTot = np.hstack([Input, InpMOR])
        if not resCalc:
            InpMOR = np.tile(MORinpNN, reps=[np.sum(biDof),1])
            biInpuTot = np.hstack([biInput, InpMOR])
        else:
            biInpuTot = []                                          # not required for residual calculations
        
        # Update the training data and return:
        tData.updateData(InpuTot, biInpuTot, biLabel, gcoef, sourceVal, diff, vel, MORinpNN)
        return tData



    def MORargExtract(self, batch, MORdiscArg, defArg=None):
        """
        Extract the relevant arguments for MOR functions.
        
        Inputs:
            batch[int]: current batch number
            MORdiscArg [list]: list of discretized variable arguments for MOR
            defArg: default argument to be passed to functions that are not part
                of the MOR:
                None: function will not be recomputed and the stored data is used
                []: function will be computed but without extra MOR arguments
                Note: this argument is not part of user input but code and it 
                      should be used in agreement with PDEinpData() and biTrainData()
                
        Outputs:
            biArg: arguments to be sent to biTrainData() function
            inpArg: arguments to be sent to PDEinpData() function
            inpNN: extra inputs to the NN due to MOR parameters
        """
        # Data:
        PDE = self.PDE
        BCtype = PDE.BCtype
        bIndNum = PDE.domain.bIndNum
        funInd = PDE.MORfunInd                                  # mapping between PDE data and MOR functions
        
        MORvar = PDE.MORvar
        argNames = MORvar.ArgNames                              # names of variable arguments for each function
        
        argInd = self.fixData.MORargInd                         # index combinations of all functions (batch loop)
        
        inpNN = []                                              # corresponding MOR input to the NN
        
        
        # Boundary-initial data:
        biArg = []
        if funInd['biData']:
            BCind = funInd['BCs']
            for bInd in range(bIndNum):                         # loop over boundary conditions
                if not BCtype[bInd]=='Dirichlet' or uf.isnone(BCind[bInd]):
                    biArg.append(None)
                    continue
                
                # Avoid recomputation if the argument for current function has not changed:
                find = BCind[bInd]                              # MOR function corresponding to currrent BC function 'g'
                ind = argInd[batch, find]                       # argument index for function specified by find
                values = MORdiscArg[find][ind,:]
                if batch>0:
                    indp = argInd[batch-1, find]                # argument index for function specified by find
                    valprev = MORdiscArg[find][indp,:]          # previous argument
                    if uf.l2Err(values, valprev)<1.e-10:
                        biArg.append(None)
                        continue
                    
                # Construct the dictionary of variable arguments:
                biArg.append(uf.buildDict(argNames[find], values))
                inpNN.extend(values)
            
            # Initial condition
            if not uf.isnone(funInd['IC']):
                find = funInd['IC']
                ind = argInd[batch, find]                       # argument index for function specified by find
                values = MORdiscArg[find][ind,:]
                
                # Avoid recomputation if the argument for current function has not changed:
                if batch==0:
                    biArg.append(uf.buildDict(argNames[find], values))
                    inpNN.extend(values)
                else:
                    indp = argInd[batch-1, find]                # argument index for function specified by find
                    valprev = MORdiscArg[find][indp,:]          # previous argument
                    if uf.l2Err(values, valprev)<1.e-10:
                        biArg.append(None)
                    else:
                        biArg.append(uf.buildDict(argNames[find], values))
                        inpNN.extend(values)
                
        else:
            biArg = []                                          # assign the default if MOR does not involve boudary-initial data
        
        
        # PDE input data:
        if funInd['inpData'] and not uf.isnone(funInd['diff']):
            find = funInd['diff']
            ind = argInd[batch, find]                           # argument index for function specified by funInd['diff']
            values = MORdiscArg[find][ind,:]
            
            # Avoid recomputation if the argument for current function has not changed:
            if batch==0:
                diffArg = uf.buildDict(argNames[find], values)
                inpNN.extend(values)
            else:
                indp = argInd[batch-1, find]                    # argument index for function specified by find
                valprev = MORdiscArg[find][indp,:]              # previous argument
                if uf.l2Err(values, valprev)<1.e-10:
                    diffArg = defArg
                else:
                    diffArg = uf.buildDict(argNames[find], values)
                    inpNN.extend(values)
        else:
            diffArg = defArg                                    # assign default argument of 'diff' is not part of MOR
            
        if funInd['inpData'] and not uf.isnone(funInd['vel']):
            find = funInd['vel']
            ind = argInd[batch, find]                           # argument index for function specified by funInd['vel']
            values = MORdiscArg[find][ind,:]
            
            # Avoid recomputation if the argument for current function has not changed:
            if batch==0:
                velArg = uf.buildDict(argNames[find], values)
                inpNN.extend(values)
            else:
                indp = argInd[batch-1, find]                    # argument index for function specified by find
                valprev = MORdiscArg[find][indp,:]              # previous argument
                if uf.l2Err(values, valprev)<1.e-10:
                    velArg = defArg
                else:
                    velArg = uf.buildDict(argNames[find], values)
                    inpNN.extend(values)
        else:
            velArg = defArg                                     # assign default argument of 'vel' is not part of MOR
            
        if funInd['inpData'] and not uf.isnone(funInd['source']):
            find = funInd['source']
            ind = argInd[batch, find]                           # argument index for function specified by funInd['source']
            values = MORdiscArg[find][ind,:]
            
            # Avoid recomputation if the argument for current function has not changed:
            if batch==0:
                sourceArg = uf.buildDict(argNames[find], values)
                inpNN.extend(values)
            else:
                indp = argInd[batch-1, find]                    # argument index for function specified by find
                valprev = MORdiscArg[find][indp,:]              # previous argument
                if uf.l2Err(values, valprev)<1.e-10:
                    sourceArg = defArg
                else:
                    sourceArg = uf.buildDict(argNames[find], values)
                    inpNN.extend(values)
        else:
            sourceArg = defArg                                  # assign default argument of 'source' is not part of MOR
            
        if funInd['inpData']:
            inpArg = [diffArg, velArg, sourceArg]               # group the arguments together
        else:
            inpArg = []                                         # assign default argument if PDE-data is not part of MOR
            
            
        inpNN = reshape(inpNN, [1, size(inpNN)])                # store the NN input in a row
        return biArg, inpArg, inpNN
        
                

    def splitLoss(self, tData, fixData=None, MORdiscArg=None, W=None):
        """
        Function to compute the components of the loss function for a given 
        input data and MOR discretization. The 'W' argument defines arbitrary 
        combination of weights on individual terms of the loss function.
        
        Inputs:
            fixData: global fixed data used throughout the class
            tData: instance of 'ManageTrainData' class
            MORdiscArg: discretization of the MOR arguments
            W [nx3]: each row corresponds to a combination of weights
        """
        # Error handling:
        if not (uf.isnone(W) or type(W)==np.ndarray):
            raise ValueError('\'W\' must be an array with shape (,3)!')
            
        # If no weight is supplied, get contribution of each term to loss function:
        if uf.isnone(W): W = np.eye(3)
        
        # Data:
        if uf.isnone(fixData):
            fixData = self.fixData
        MORbatchNum = fixData.MORbatchNum
        lossVecflag = fixData.lossVecflag
        tfData = self.tfData
        if uf.isnone(MORdiscArg): MORdiscArg = fixData.MORdiscArg
        
        lossComp = 0.
        if lossVecflag: lossVec = []
        else:           lossVec = None
        for batch in range(MORbatchNum):                                                # loop over MOR batches
            tData = self.trainData(batch, MORdiscArg, tData)
            BCloss, ICloss, varLoss, lossVecTmp = tData.splitLoss(tfData, lossVecflag)  # individual loss components
            lossComp += np.array([[BCloss, ICloss, varLoss]]).T                         # column vector
            if lossVecflag: lossVec.append(lossVecTmp)                                  # append loss field for current batch
        lossVal = np.matmul(W, lossComp)                                                # apply the weights
    
        return lossVal, tData, lossVec



    def trainWeight(self, weight, tData, MORdiscArg, normalizeW, useOriginalW, lossTot=1.e6):
        """
        Function to determine the penalty weights for training. The function 
        adjusts the weights so that different terms have contributions specified
        by 'weight'. The maximum value of objective is set to 1.e6 so that
        there is a general sense of convergence for the training process.
        
        Output:
            tData: instance of training data management class passed back to save
                computation on PDE input data
            normalizeW: wether normalize the weights so that individual terms have
                weights specified by 'weight' argument at the initiation of iterations
            useOriginalW: if True use 'weight' directly as training weights 'trainW'
        """
        # Data:
        timeDependent = self.PDE.timeDependent
        trainRes = self.trainRes
        
        # Compute the individual loss terms:
        lossVal, tData, _ = self.splitLoss(tData, MORdiscArg=MORdiscArg)
        lossVal = reshape(lossVal,3)
        if not timeDependent:
            lossTmp = [lossVal[0], lossVal[2]]      # remove initial condition
        else:
            lossTmp = lossVal

        # Compute the training weights:
        if useOriginalW:
            trainW = weight
        elif normalizeW:
            nw = len(weight)
            W = np.tile(weight, [nw, 1])
            W = W/reshape(weight, [nw, 1])
            W = np.sum(W, axis=1, keepdims=True)*uf.vstack(lossTmp)
            trainW = reshape(lossTot/W, nw)
        else:
            loss = np.sum(np.array(weight)*np.array(lossTmp))
            trainW = lossTot/loss*np.array(weight)
        if not timeDependent: trainW = np.array([trainW[0], 0., trainW[1]])
        
        trainW = np.array(trainW)
        
        # Print weight data to output and case file:
        string  = 'Training weight information:\n'
        string += '\tboundary condition loss value: ' + np.array2string(lossVal[0], precision=4) + '\n'
        string += '\tinitial condition loss value: ' + np.array2string(lossVal[1], precision=4) + '\n'
        string += '\tintegral loss value: ' + np.array2string(lossVal[2], precision=4) + '\n'
        string += '\trequested weight on each term: ' + str(weight) + '\n'
        string += '\tcorresponding training weights: ' + np.array2string(trainW, precision=4) + '\n\n'
        print(string)
        trainRes.writeCase(string)
        
        return trainW, tData, lossVal



    def weightUpdate(self, trainW, tData, MORdiscArg):
        """
        Function to periodically update the training weights to ensure that 
        the weights of individual terms in the loss function stay constant.
        The desired weight is preset to [10, 10, 1] meaning that the weighted
        loss value of boundary-initial terms is kept ten times higehr than the
        integral term. If this balance is violated, the function incrementally
        adds 10% to smaller weights so that after a set of weight updates the 
        balance is restored.
        
        Output:
            trainW: current training weight values
            tData: instance of training data management class passed back to save
                computation on PDE input data
        """
        
        # Data:
        timeDependent = self.PDE.timeDependent
        trainRes = self.trainRes
        
        # Compute the individual loss terms and total loss:
        lossVal, tData, _ = self.splitLoss(tData, MORdiscArg=MORdiscArg)
        lossVal = np.array(lossVal)
        lossTot = np.sum(trainW*lossVal)                # total loss value
        if not timeDependent:
            weight = np.array([10, 1.0])                # desired weights
            lossVal = lossVal[[0,2]]                    # remove initial condition
        else:
            weight = np.array([10, 10, 1.0])            # desired weights
            
        effLoss = trainW*weight*lossVal                 # effective loss
        wBar = effLoss/min(effLoss)                     # normalize the weights
        wBar = 0.1*(wBar>2.0) + 1                       # add 10% to under-represented terms
        trainW = wBar*trainW                            # update the weights
        alpha = lossTot/np.sum(trainW*lossVal)          # coefficent to reset the total loss to current value
        trainW = alpha*trainW                           # reset the total loss to current value
        if not timeDependent: trainW = np.array([trainW[0], 0., trainW[1]])
        
        # Print weight data to output and case file:
        string = '\nupdated training weights:: ' + np.array2string(trainW, precision=4) + '\n'
        print(string)
        trainRes.writeCase(string)
        
        return trainW, tData
    


    def train(self, folderpath, weight=None, smpScheme='uniform', epochNum=500000,
              tol=1.e-1, verbose=True, saveFreq=100, pltReplace=True, saveMORdata=False,
              frac=None, addTrainPts=True, suppFactor=1.0, multiTrainUpd=False,
              trainUpdelay=2e4, tolUpd = 0.01, reinitrain = True,
              updateWeights=False, normalizeW=False, adjustWeight=False, useOriginalW=False,
              batchNum=None, batchLen=None, shuffleData=False, shuffleFreq=1):
        """
        Fuction to train the VarNet. This function is the only one to run its 
        own session since here we prefer efficiency above clarity.
        
        Inputs:
            folderpath: path to a folder to store the training data.
            weight: penalty weights for loss function terms, including the 
                boundary, initial, and integral terms
            smpScheme: sampling scheme
                uniform: constant uniform samples
                random: randomly sample the space-time with a uniform distribution
                optimal: use feedback from PDE-residual to select optimal training points
            epochNum: number of training epochs
            tol: stopping tolerance for objective (max objective value: 1.e6)
            verbose: output training results if True
            saveFreq: frequency of saving and reporting the training results
            pltReplace: if True replace the training result plots, o.w., keep old plots
            saveMORdata: save data corresponding to MOR argument combinations for faster training
            frac: for non-uniform sampling determines the fraction that is drawn non-uniformly
            addTrainPts: if True add optimal training points when 'smpScheme=optimal',
                i.e., refine the mesh, o.w., keep the total number of training points constant
                Note: multiple loops of optimal sampling only changes the location of 
                    optimal samples and does not generate new samples indefinitely
            suppFactor: if optimal training points are added to exisiting training
                points, this factor scales the support of these new optimal test functions
                i.e., each dimension is multiplied by 'suppFactor<1.0'
            multiTrainUpd: training points are preiodically updated if True
            trainUpdelay: minimum number of epochs allowed before updating training points
            tolUpd: tolerance to add non-uniform training point
            reinitrain: if True reinitialize training variables after addition 
                        of new trainin points (often otherwise optimization gets stuck)
            updateWeights: if True, preiodically update the training weights to 
                ensure that individual terms keep having a desired balance in the objective
            normalizeW: wether normalize the weights so that individual terms have
                weights specified by 'weight' argument at the initiation of iterations
            adjustWeight: increase weight on BICs after adding non-uniform samples
            useOriginalW: if True use 'weight' directly as training weights 'trainW'
            batchNum: number of batches used for training
            batchLen: length of training batches (recommended: 32-512 for ADAM optimizer)
            shuffleData: shuffle data before dividing them into batches
            shuffleFreq: shuffle data every 'shuffleFreq' epochs
        """
        # Error handling:
        if uf.isnone(folderpath) or uf.isempty(folderpath):
            raise ValueError('a folder path must be provided to backup the trained model!')
        self.folderpath = folderpath                                # store folder path for later use
        
        PDE = self.PDE
        timeDependent = PDE.timeDependent
        if uf.isnone(weight) and timeDependent:
            weight = [1., 1., 1.]
        elif uf.isnone(weight):
            weight = [1., 1.]
        elif not uf.isnone(weight) and timeDependent and not len(weight)==3:
            raise ValueError('weight dimension does not match!')
        elif not uf.isnone(weight) and not timeDependent and not len(weight)==2:
            raise ValueError('weight dimension does not match!')
        
        if not (smpScheme=='uniform' or smpScheme=='random' or smpScheme=='optimal'):
            raise ValueError('sampling scheme is not valid!')
        self.smpScheme = smpScheme
        
        if uf.isnone(frac):
            if addTrainPts: frac=0.50
            else:           frac=0.25
        
        if not addTrainPts and np.abs(suppFactor-1.0)>1.e-15:
            warnings.warn('\'suppFactor\' is set to 1.0 since the number of training points does not change!')
            suppFactor = 1.0
            
        if uf.isnone(batchNum) and uf.isnone(batchLen) and shuffleData:
            warnings.warn('shuffling data is possible for batch-optimization, setting \'shuffleData\' to False!')
            shuffleData = False
        
        # Store local variables (input arguments to this function) to be written to case file:
        argDict = locals()
        
        # Load data:
        modelId = self.modelId
        MORvar = self.PDE.MORvar                                    # MOR class instance for the PDE
        
        self.fixData.setFEdata()                                    # generate FE data corresponding to initial uniform sampling
        fixData = self.fixData
        integPnum = fixData.integPnum
        MORbatchNum = fixData.MORbatchNum                           # number of MOR batches
        MORbatchRange = np.arange(MORbatchNum)
        
        tfData = self.tfData
        graph = tfData.graph
        saver = tfData.saver
        sess = tfData.sess
        
        # Initialize tensorflow variables:
#        with graph.as_default():
#            sess.run(tf.global_variables_initializer())             # initialize all variables

        # Training data:
        Input, InputRNN, biInput, biDof = self.trainingPoints()     # first set of training points are always uniformly sampled
        if uf.isnone(MORvar):
            MORdiscArg = None
            saveMORdata = False
        else:
            MORdiscScheme = self.MORdiscScheme                      # discretization scheme for MOR variables
            MORdiscArg = MORvar.discretizeArg(MORdiscScheme)        # discretize the arguments
        if modelId=='RNN': Input = InputRNN
        tData = ManageTrainData(Input, biInput, batchNum, batchLen, # instance of training data management class
                                saveMORdata, MORbatchNum)
        
        # Construct an instance of TrainResult class to manage training results:
        if not uf.isnone(fixData.cEx):
            cExFlg = True
        else:
            cExFlg = False
        trainRes = TrainResult(folderpath, cExFlg, verbose, saveFreq, pltReplace)
        trainRes.initializeCase(self, argDict)                      # write training data to text file for later reference
        self.trainRes = trainRes
        
        # Weight determination:
        trainW, tData, lossVal = self.trainWeight(weight, tData, MORdiscArg, normalizeW, useOriginalW)
#        if not uf.isnone(MORvar): tData.inputUpdated = True         # use already computed PDE-data if possible
        trainRes.trainWeight = trainW
        tData.updateDictFields('trainW', trainW)                    # update the weight in training data
        trainRes.lossComp.append(lossVal)                           # store the initial loss cmponents
        
        # Store initial uniform input data for universal cost comparison:
        tData0 = tData
        if smpScheme=='optimal' and addTrainPts:
            fixData0 = FIXData(self, integPnum)                     # create an independent fixed data object for universal cost comparison
            fixData0.setFEdata()                                    # FE data corresponding to initial uniform training points
            fixData0.removeInputData()                              # remove uniform input data to minimize memory usage
        else:
            fixData0 = None
        
        # Trainig variables:
        min_loss = float('inf')
        epoch_time = 0
        tp_epoch = 1
        tp_updates = 0
        resVal, err, lossComp, lossVec = [None]*4
        filepath2 = os.path.join(folderpath, 'best_model')
        
        # =================================================================== #
        # Training loop:
        for epoch in range(1,epochNum+1):                               # loop over training epochs
            t = time.clock()
            current_loss = 0
                
            for batch in MORbatchRange:                                 # loop over MOR batches
                tData = self.trainData(batch, MORdiscArg, tData)
                current_loss += tData.optimIter(tfData)
            epoch_time += time.clock()-t
            
            if shuffleData and epoch%shuffleFreq==0:
                tData.shuffleTrainData(fixData)                         # shuffle training points
            # do not shuffle MOR batches and shuffle training points once for all MOR batches.
            
            if epoch%saveFreq==0:
                if min_loss>current_loss:
                    min_loss = current_loss
                    saver.save(sess, filepath2, global_step=epoch)      # save best model so far
                resVal, _, err, _ = self.residual()
#                if not uf.isnone(MORvar): tData0.inputUpdated = True    # use already computed PDE-data if possible
                lossComp, _, lossVec = self.splitLoss(tData0, fixData0)
                
            # Output training results:
            trainRes.iterOutput(epoch, current_loss, min_loss, epoch_time, 
                                resVal, err, lossComp, lossVec)
            
            # Weight update:
            if updateWeights and (epoch-tp_epoch)>=(trainUpdelay-1) and epoch%1000==0:
                trainW = self.weightUpdate(trainW, tData, MORdiscArg)
                if not uf.isnone(MORvar): tData.inputUpdated = True     # use already computed PDE-data if possible
                tData.updateDictFields('trainW', trainW)                # update the weight in training data
                
            # Convergence:
            if current_loss<tol:
                string = 'Training completed!'
                print(string)
                trainRes.writeCase(string)
                break
            
            # Regenerate the training data:
            if not smpScheme=='uniform' and (multiTrainUpd or tp_updates==0) and (epoch-tp_epoch)>=(trainUpdelay-1):
                t_loss = np.array(self.trainRes.loss[-5:])
                tp_conv = t_loss[:-1]-t_loss[1:]
                tp_conv = np.sum(tp_conv[tp_conv>0])
                if tp_conv/t_loss[-1]<tolUpd:                           # if iterations have converged
                    min_loss = float('inf')                             # reset the best model loss value
                    tp_epoch = epoch                                    # reset training points update epoch
                    tp_updates += 1                                     # increment training points update counts
                    trainRes.inpIter.append(epoch)                      # store the corresponding epoch
                    
                    Input, _, biInput, biDof = self.trainingPoints(smpScheme, frac, addTrainPts, suppFactor)
                    if addTrainPts: fixData = self.fixData              # update the fixed training data
                    tData = ManageTrainData(Input, biInput,             # space-time discretization updated
                                            batchNum, batchLen, saveMORdata, MORbatchNum)
                    
                    string = '\n\n==========================================================\n'
                    string += 'Training points updated.\n\n'
                    print(string)
                    trainRes.writeCase(string)
                    
                    if reinitrain:
                        string = 'trainable variables reinitialized.\n\n'
                        print(string)
                        trainRes.writeCase(string)
                        
                        # Initialize tensorflow variables:
                        with graph.as_default():
                            sess.run(tf.global_variables_initializer())    # initialize all variables
                    
                    # Weight determination:
                    if adjustWeight:
                        wInteg = weight[-1]
                        weight = [5*w for w in weight[:-1]]                 # add more weights on BICs
                        weight.append(wInteg)                               # append the integral weight back
                    trainW, tData, _ = self.trainWeight(weight, tData, MORdiscArg, normalizeW, useOriginalW)
#                    if not uf.isnone(MORvar): tData.inputUpdated = True     # use already computed PDE-data if possible
                    tData.updateDictFields('trainW', trainW)                # update the weight in training data
                    

            
        
    def loadModel(self, iterNum=None, folderpath=None):
        """
        Function to restore the training data to a stored checkpoint.
        
        Note: there are four types of checkpoint files:
            - meta: stores the computational graph and is identical across checkpoints
            - index: stores the metadata for each variable in the graph
            - data: stores the values of variables at each checkpoint
            - checkpoint: text file with addresses of the saver files
                          (should be edited to store different checkpoints)
        
        Inputs:
            iterNum: number of iteration for which the checkpoint should be restored
            folderpath: path to the folder that contains the checkpoints
        """
        
        # Error handling:
        if not hasattr(self, 'trainRes') and uf.isnone(folderpath):
            raise ValueError('\'folderpath\' must be provided!')
        elif uf.isnone(folderpath):
            folderpath = self.trainRes.folderpath
        else:
            trainRes = TrainResult(folderpath)
            trainRes.loadData()
            plotpath = os.path.join(folderpath, 'plots')
            if not os.path.exists(plotpath):
                os.makedirs(plotpath)
            trainRes.plotpath = plotpath
            self.trainRes = trainRes
        
        # Load tensorFlow variables:
        tfData = self.tfData
        sess = tfData.sess          
        graph = tfData.graph
        
        # Function to create the relevant checkpoint file:
        check_path = os.path.join(folderpath, 'checkpoint')
        def checkpoint(modelid):
            """Construct the file paths and write the checkpoint file."""
            modelpath = repr(os.path.join(folderpath, 'best_model-' + str(modelid)))
            string = 'model_checkpoint_path: ' + modelpath
            string += '\nall_model_checkpoint_paths: ' + modelpath + '\n'
            with open(check_path, 'w') as myfile: myfile.write(string)
        
        # Extract the stored iteration numbers:
        if uf.isnone(iterNum):
            iterNum = []
            for file in os.listdir(folderpath):
                if '.index' in file:
                    ind1 = file.rfind('-') + 1
                    ind2 = file.rfind('.')
                    iNum = int(file[ind1:ind2])
                    iterNum.append(iNum)
            iterNum.sort(reverse=True)
        else:
            iterNum = [iterNum]             # convert to list for consistency
        
        # Load the desired model from folder:
        meta_path = os.path.join(folderpath, 'best_model-' + str(iterNum[-1]) + '.meta')
        errFlg = True
        for iNum in iterNum:
            # Check if all relevant checkpoint data are available:
            filename = 'best_model-' + str(iNum)
            filepath = os.path.join(folderpath, filename + '.index')
            if not os.path.isfile(filepath): continue
            filepath = os.path.join(folderpath, filename + '.data-00000-of-00001')
            if not os.path.isfile(filepath): continue
                
            # Restore the data:
            checkpoint(iNum)        # create the appropriate checkpoint data
            with graph.as_default():
                saver = tf.train.import_meta_graph(meta_path, clear_devices=True)
                saver.restore(sess, tf.train.latest_checkpoint(folderpath))
            
            errFlg = False
            break
            
        if errFlg: raise ValueError('no restorable checkpoint data found!')
        
        # Save data:
        tfData.saver = saver



    def evaluate(self, x=None, t=None, batch=None, MORarg=None):
        """
        Function to approximate the solution of the PDE via the NN.
        
        Inputs:
            x [n x dim]: vector of spatial discretizations
            t [n x 1]: corresponding vector of temporal coordinates
            batch [int]: index of the batch of MOR parameters
            MORarg [n x (inpDim-feDim)]: corresponding vector of MOR arguments
        """
        
        # Data:
        dim = self.dim
        modelId = self.modelId
        
        PDE = self.PDE
        timeDependent = PDE.timeDependent
        MORvar = PDE.MORvar                                             # MOR class instance for the PDE
        
        fixData = self.fixData
        feDim = fixData.feDim
        MORbatchNum = fixData.MORbatchNum                                  # number of MOR batches
        MORdiscArg = fixData.MORdiscArg                                 # discretization of MOR arguments
        
        # TensorFlow model data:
        tfData = self.tfData
        inpDim = tfData.inpDim
        
        # Error handling:
        if uf.isnone(x):
            if (timeDependent and (uf.isnone(t) or modelId=='RNN')) or not timeDependent:
                dof = fixData.nt
                Input = fixData.uniform_input                           # extract the whole input
            else:
                dof = fixData.dof
                x = fixData.uniform_input[:dof, :dim]                   # only extract the spatial discretization
        elif not shape(x)[1]==dim:
            raise ValueError('spatial coordinates dimension does not match domain!')
        else:
            dof = shape(x)[0]
            
        if timeDependent and not uf.isnone(t) and not (size(t)==1 or shape(t)[0]==dof):
            raise ValueError('temporal discretrization does not match spatial discretization!')
        elif timeDependent and not modelId=='RNN' and not uf.isnone(t) and size(t)==1:
            t = t*np.ones([dof,1])
        
        if not uf.isnone(MORvar) and uf.isnone(batch) and uf.isnone(MORarg):
            raise ValueError('batch number or argument values must be given for MOR!')
        elif not uf.isnone(MORvar) and uf.isnone(batch) and not shape(MORarg)[1]==inpDim-feDim:
            raise ValueError('MOR argument dimension does not match the NN input size!')
        elif not uf.isnone(MORvar) and uf.isnone(batch) and not (shape(MORarg)[0]==1 or shape(MORarg)[0]==dof):
            raise ValueError('MOR argument number does not match \'x\' dimension!')
        elif not uf.isnone(MORvar) and not uf.isnone(batch) and uf.isnumber(batch) and batch>MORbatchNum-1:
            raise ValueError('requested batch number is higher than total available batches!')
        elif not uf.isnone(MORvar) and uf.isnone(batch) and shape(MORarg)[0]==1:
            MORarg = np.tile(MORarg, [dof,1])
        
        if modelId=='RNN': RNNdata = self.RNNdata
        
        # Construct the input:
        if modelId=='RNN':
            Input = RNNdata.buildInput(x)
        elif timeDependent and not uf.isnone(t):
            Input = np.concatenate([x, t], axis=1)
        elif not uf.isnone(x):
            Input = x
            
        # Add MOR arguments:
        if not uf.isnone(MORvar) and uf.isnone(batch):
            tData = ManageTrainData(Input, biInput=[])
            InpuTot = np.hstack([Input, MORarg])
            tData.updateData(InpuTot=InpuTot)
        else:
            tData = ManageTrainData(Input, biInput=[])
            tData = self.trainData(batch, MORdiscArg, tData, resCalc=True)
        
        # Evaluate the NN:
        cApp = tData.runSession(['model'], tfData)[0]
        
        # Extract the relevant RNN output:
        if modelId=='RNN':
            cApp = RNNdata.flatten(cApp)                                # flatten the output
            tDiscIND = RNNdata.timeIndex(len(x), t)                     # extract relevant time indices
            cApp = cApp[tDiscIND,:]                                     # extract the corresponding values
            
        return cApp



    def residual(self, Input=None, tDiscIND=None, batch=None):
        """
        Function to compute the residual of the PDE.
        
        Inputs:
            Input: input to NN (also used to compute the exact solution)
            tDiscIND: determines the indices of entries to the RNN that are of 
                interest (the rest of the output is discarded)
            batch: batch number for which the residual should be calculated
        """
        
        # Data:
        noInput = uf.isnone(Input)
        dim = self.dim
        modelId = self.modelId
        
        PDE = self.PDE
        timeDependent = PDE.timeDependent
        d_diffFun = PDE.d_diffFun
            
        if modelId=='RNN':
            RNNdata = self.RNNdata
            if not noInput:
                Input = RNNdata.flatten(Input)                          # flatten the input for field computations
                if uf.isnone(tDiscIND):
                    tDiscIND = np.ones(len(Input), dtype=bool)
        
        fixData = self.fixData
        hVec = fixData.hVec
        elemSize = np.prod(hVec)
        MORbatchNum = fixData.MORbatchNum                                   # number of MOR batches
        MORdiscArg = fixData.MORdiscArg                                  # discretization of MOR arguments
        if uf.isnumber(batch) and batch>MORbatchNum-1:
            raise ValueError('requested batch number is higher than total available batches!')
        elif uf.isnumber(batch):
            batchRange = range(batch,batch+1)
            MORbatchNum = 1
        elif uf.isnone(batch):
            batchRange = range(MORbatchNum)
        
        if noInput:
            Input = fixData.uniform_input
            cEx = fixData.cEx
            if modelId=='RNN':
                Input = RNNdata.flatten(Input)                          # flatten the default input for field computations
                tDiscIND = RNNdata.tDiscIND
        elif not uf.isnone(PDE.cEx) and timeDependent:
            cEx = PDE.cEx(Input[:, :dim], Input[:, dim:dim+1])
        elif not uf.isnone(PDE.cEx):
            cEx = PDE.cEx(Input[:, :dim])
        
        # TensorFlow variables:
        tfData = self.tfData
        
        # Gradient of the diffusivity field (needs to be moved into trainData() function):
        if noInput:
            diff_dx = fixData.d_diff
        elif timeDependent:
            diff_dx = d_diffFun(Input[:,:dim], Input[:, dim:dim+1])
        else:
            diff_dx = d_diffFun(Input[:,:dim])
        
        # Pre-process the input for the RNN:
        if modelId=='RNN':
            Input = RNNdata.fold(Input)                                 # fold the input for RNN computations
        
        # Compute the AD-PDE residual:
        res = 0
        err = 0
        tData = ManageTrainData(Input, biInput=None)
        for batch in batchRange:                                        # loop over MOR batches
            tData = self.trainData(batch, MORdiscArg, tData, resCalc=True)
            (cApp, resVec) = tData.runSession(['model', 'residual'], tfData, diff_dx=diff_dx)
            
            # Post-processing for the RNN:
            if modelId=='RNN':
                cApp = RNNdata.flatten(cApp)
                cApp = cApp[tDiscIND,:]
                cEx = cEx[tDiscIND,:]
                resVec = resVec[tDiscIND,:]
            
            if not uf.isnone(PDE.cEx):
#                err += np.sqrt(np.sum((cEx-cApp)**2)*elemSize)          # intetgration over domain
                err += uf.l2Err(cEx, cApp)
            else:
                err = None
            res += np.sqrt(np.sum(resVec**2)*elemSize)                  # intetgration over domain
        
        # Average the values for all batches:
        res = res/MORbatchNum
        if not uf.isnone(PDE.cEx):
            err = err/MORbatchNum
        
        return res, resVec, err, cApp
        
        
        
    def optTrainPoints(self, frac=0.25, addTrainPts=True, suppFactor=1.0):
        """
        Function to generate the training points.
        
        Input:
            frac: fraction of points to be selected via optimal sampling
            addTrainPts: if True add optimal training points when 'smpScheme=optimal',
                i.e., refine the mesh, o.w., keep the total number of training points constant
            suppFactor: support scaling for optimal training points if they are 
                added to already exisiting training points
        """
        # Data:
        dim = self.dim
        discNum = self.discNum
        bDiscNum = self.bDiscNum
        
        PDE = self.PDE
        timeDependent = PDE.timeDependent
        domain = PDE.domain
        
        fixData = self.fixData
        feDim = fixData.feDim
        integNum = fixData.integNum
        nt = fixData.nt0
        nT = fixData.nT
        hVec = fixData.hVec
        delta = fixData.delta
        
        
        # Fraction of samples that are drawn uniformly across dimensions:
        if addTrainPts:
            frac2 = 1                                                           # keep uniform samples fixed
        else:
            frac2 = (1-frac)**(1/feDim)
        
        # Get the mesh for the time coordinate:
        if timeDependent:
            tDiscNum2 = math.ceil(frac2*self.tDiscNum)
            _, t_coord = self.timeDisc(tDiscNum2)
        else:
            tDiscNum2 = 1
            t_coord = []
        
        # Get the mesh for the domain:
        discNum2 = [math.ceil(frac2*disc2) for disc2 in discNum]
        mesh = domain.getMesh(discNum2, bDiscNum)
        dof = mesh.dof
        coord = mesh.coordinates                                                # mesh coordinates for inner domain
        
        # Pair samples in space-time:
        input2 = uf.pairMats(coord, t_coord)
        
        
        # Generate the optimal samples using rejection sampling (see Wikipedia):
        if addTrainPts:                                                         # number of optimal samples
            nt1 = math.ceil(frac*nt)                                            # add optimal samples
        else:
            nt1 = nt - dof*tDiscNum2                                            # keep total number of samples constant
        
        if np.abs(suppFactor-1.0)<1.e-15: tolt, tole = [None]*2
        else:                                                                   # adjust discretization tolerance if added points have smaller support
            tole = suppFactor*hVec[:dim]
            if timeDependent: tolt = suppFactor*hVec[-1]
        
        def resfun(inpuT=None):
            """Residual function to generate the samples in inner-domain according to."""
            _, resVec, _, _ = self.residual(inpuT)
            return np.abs(resVec)
        
        def smpfun():
            # Mesh over time coordinate:
            if timeDependent:
                _, t_coord = self.timeDisc(rfrac=1, sortflg=False, discTol=tolt)# random drawings over time
            else: t_coord = []
            
            # Mesh for the inner-domain:
            mesh = domain.getMesh(discNum, bDiscNum, rfrac=1, sortflg=False,    # random drawings over space
                                  discTol=tole)
            coord = mesh.coordinates                                            # mesh coordinates for inner domain
            
            return uf.pairMats(coord, t_coord)                                  # pair samples in space-time
            
        input1 = uf.rejectionSampling(resfun, smpfun, nt1)                      # rejection sampling algorithm
        
        # Update total number of discretization points:
        if addTrainPts:
            nt = nt1 + nt                                                       # new total number of samples
            nT = nt*integNum
        
        # Put uniform and optimal samples together and sort them according to time:
        # (do not sort if the optimal training points have different support size)
        inpuT = np.vstack([input1, input2])
        coord = inpuT[:, :dim]
        if timeDependent and np.abs(suppFactor-1.0)<1.e-15:
            t_coord = inpuT[:, dim:dim+1]
            ind = np.argsort(t_coord, axis=0)
            ind = reshape(ind, nt)
            coord = coord[ind]
            t_coord = t_coord[ind]
        elif timeDependent:
            t_coord = inpuT[:, dim:dim+1]
        
        
        # Initial and boundary training data:
        biInput, biDof, biInput1, biInput2 = self.optBiTrainPoints(frac, addTrainPts)
        
        
        # Plot samples:
        if dim==1 or (dim==2 and not timeDependent):
            def resFun(x,t=None):
                """Residual plot function handle."""
                if timeDependent:
                    if size(t)==1: t = t*np.ones([len(x),1])
                    Input = np.hstack([x,t])
                else:
                    Input = x
                _, resVec, _, _ = self.residual(Input)
                return np.abs(resVec)
            
            # Contour plot:
            contPlot = ContourPlot(domain, PDE.tInterval)
            contPlot.conPlot(resFun)
            
            # Save the plot:
            plotpath = self.trainRes.plotpath
            epoch = self.trainRes.inpIter[-1]
            filename = '{}'.format(epoch) + '_resField'
            filepath = os.path.join(plotpath, filename + '.png')                # image of the plot
            plt.savefig(filepath, dpi=300)
            filepath = os.path.join(plotpath, filename + '.eps')                # eps file
            plt.savefig(filepath, dpi=300)
            
            # Add training points:
            plt.plot(input2[:,0], input2[:,1], 'y.', markersize=1)
            plt.plot(input1[:,0], input1[:,1], 'w.', markersize=4)              # optimal points
            plt.plot(biInput2[:,0], biInput2[:,1], 'y.', markersize=1)
            plt.plot(biInput1[:,0], biInput1[:,1], 'w.', markersize=4)          # optimal boundary points
            
            # Save the plot:
            filename = '{}'.format(epoch) + '_optimalSmp'
            filepath = os.path.join(plotpath, filename + '.png')                # image of the plot
            plt.savefig(filepath, dpi=300)
            filepath = os.path.join(plotpath, filename + '.eps')                # eps file
            plt.savefig(filepath, dpi=300)
            plt.show()
        
        
        # Integration points:
        Coord = np.zeros([nT,dim])                                              # store spatial coordinates of integration points
        he = hVec[:dim]                                                         # element sizes
        if np.abs(suppFactor-1.0)<1.e-15:
            suppScale = 1.0
        else:
            suppScale = np.ones([nt,1])
            suppScale[:nt1,:] = suppFactor*suppScale[:nt1,:]
        for d in range(dim):                                                    # add integration points to spatial coordinates of each training point
            coordTmp = reshape(coord[:,d], [nt, 1]) + he[d]*delta[d,:]*suppScale
            Coord[:,d] = np.reshape(coordTmp, nT)                               # re-arrange into a column
            
        if timeDependent:                                                       # add integration points to time-coordinate of each node
            ht = hVec[-1]
            tCoord = t_coord + ht*delta[-1,:]*suppScale
            tCoord = np.reshape(tCoord, [nT, 1])                                # re-arrange into a column
            Input = np.concatenate([Coord, tCoord], axis=1)
        else:
            Input = Coord
        
        # Update the fixed data if the number of training points has changed:
        if addTrainPts:
            self.fixData.updateOptimData(frac, suppFactor)
        
        InputRNN = []                                                           # redundant output returned for global compatibility
        return Input, InputRNN, biInput, biDof



    def optBiTrainPoints(self, frac=0.25, addTrainPts=True):
        """
        Function to generate optimal boundary-initial training points.
        
        Input:
            frac: fraction of points to be selected via optimal sampling
        """
        # Data:
        dim = self.dim
        discNum = self.discNum
        bDiscNum = self.bDiscNum
        
        PDE = self.PDE
        timeDependent = PDE.timeDependent
        domain = PDE.domain
        
        fixData = self.fixData
        feBiDim = fixData.feDim-1        # dimension of boundary-initial conditions (one less than space-time)
        biDof = fixData.biDof0
        uniform_biInput = fixData.uniform_biInput
        
        tfData = self.tfData
        model = tfData.model
        Input_tf = tfData.compTowers[0].Input
        sess = tfData.sess
        
        
        # Fraction of samples that are drawn uniformly across dimensions:
        if addTrainPts:
            frac2 = 1                                                           # keep uniform samples fixed
        else:
            frac2 = (1-frac)**(1/feBiDim)
        
        # Get the mesh for the time coordinate:
        if timeDependent:
            tDiscNum2 = math.ceil(frac2*self.tDiscNum)
            _, t_coord = self.timeDisc(tDiscNum2)
        else:
            t_coord = []
        
        # Get the mesh for the domain:
        discNum2 = [math.ceil(frac2*disc2) for disc2 in discNum]
        bDiscNum2 = math.ceil(frac2*bDiscNum)
        mesh = domain.getMesh(discNum2, bDiscNum2)
        
        # Initial and boundary training data:
        biInput2, biDof2 = self.biTrainPoints(mesh, t_coord)
        
        
        # Generate the optimal samples using rejection sampling (see Wikipedia):
        if addTrainPts:                                                             # number of optimal samples
            biDof1 = [math.ceil(frac*bidof1) for bidof1 in biDof]                   # add optimal samples
        else:
            biDof1 = np.array(biDof) - np.array(biDof2)                             # keep total number of samples constant
        
        def resfun(biInpuT=None):
            """Least-square function to generate the samples for boundary-initial conditions."""
            if uf.isnone(biInpuT):    biInpuT = uniform_biInput
            val = sess.run(model(Input_tf), {Input_tf: biInpuT})
            biLab = self.biTrainData(biInpuT, biDof)
            return (val-biLab)**2
        
        def smpfun():
            if timeDependent:
                _, t_coord = self.timeDisc(rfrac=1, sortflg=False)                  # random drawings over time
            else: t_coord = []
            mesh = domain.getMesh(discNum, bDiscNum, rfrac=1, sortflg=False)        # random drawings over space
            biInput, _ = self.biTrainPoints(mesh, t_coord)
            return biInput
            
        biInput1 = uf.rejectionSampling(resfun, smpfun, biDof1, biDof)              # rejection sampling algorithm
        
        # Update total number of discretization points on each boudary:
        if addTrainPts:
            biDofNew = np.array(biDof1) +  np.array(biDof2)                         # new total samples: biDof1 + biDof2
        else:
            biDofNew = biDof
        
        # Put uniform and optimal samples together and sort them according to time:
        biInput1new = uf.listSegment(biInput1, biDof1)
        biInput2new = uf.listSegment(biInput2, biDof2)
        biInput = []
        for i in range(len(biDof1)):
            biInput.append( uf.vstack([biInput1new[i], biInput2new[i]]) )
            if timeDependent:
                coord = biInput[i][:, :dim]
                t_coord = biInput[i][:, dim:dim+1]
                ind = np.argsort(t_coord, axis=0)
                ind = reshape(ind, biDofNew[i])
                coord = coord[ind]
                t_coord = t_coord[ind]
                biInput[i] = uf.hstack([coord, t_coord])
        biInput = uf.vstack(biInput)
        
        return biInput, biDofNew, biInput1, biInput2



    def simRes(self, batch=None, tcoord=None, plotpath=None, pltFrmt='png'):
        """
        Function to plot the simulation results.
        
        Inputs:
            batch [int]: index of the batch of MOR parameters
            tcoord [list]: time values for which the plots are generated
            plotpath: path to store the plots
            pltFrmt: plot format
        
        Note: this function needs to be extended for more general cases!
        """
        # Error handling:
        if not hasattr(self, 'trainRes') and uf.isnone(plotpath):
            raise ValueError('\'plotpath\' must be provided!')
        elif uf.isnone(plotpath):
            plotpath = self.trainRes.plotpath
            
        if not (pltFrmt=='png' or pltFrmt=='jpg' or pltFrmt=='pdf' or pltFrmt=='eps'):
            raise ValueError('invalid plot format!')
        elif uf.isnone(batch):
            pltFrmt = '.' + pltFrmt
        else:
            pltFrmt = '-b=' + str(int(batch)) + '.' + pltFrmt
            
        # Data:
        dim = self.dim
        modelId = self.modelId
        
        PDE = self.PDE
        timeDependent = PDE.timeDependent
        tInterval = PDE.tInterval
        cExact = PDE.cEx
        domain = PDE.domain
        
        if modelId=='RNN': RNNdata = self.RNNdata
        
        # Time snapshots for plotiing:
        if timeDependent and uf.isnone(tcoord):
            tcoord = np.linspace(tInterval[0], tInterval[1], num=5)
        elif not timeDependent:
            tcoord = [0.]

        # Contour plot class:
        contPlot = ContourPlot(domain, tInterval)
        
        # Function handles:
        cAppFun = lambda x, t=None: self.evaluate(x, t, batch)
        
        def resFun(x, t=None):
            """Residual plot function handle."""
            if modelId=='RNN':
                Input = RNNdata.buildInput(x)
                tDiscIND = RNNdata.timeIndex(len(x), t)
            elif timeDependent:
                Input = np.concatenate([x, t*np.ones([len(x),1])], axis=1)
                tDiscIND = None
            else:
                Input = x
                tDiscIND = None
                
            _, resVec, _, _ = self.residual(Input, tDiscIND, batch)
            return resVec
        
        # Exact solution and the corresponding error function:
        if not uf.isnone(cExact):
            if timeDependent:
                cExFun = lambda x,t: cExact(x, t*np.ones([len(x),1]))
            else:
                cExFun = lambda x,t=None: cExact(x)
            cErrFun = lambda x,t=None: cExFun(x,t)-cAppFun(x,t)
        
        # Loss field:
        if hasattr(self, 'trainRes') and not uf.isnone(self.trainRes.lossVec):
            if uf.isnone(batch): lossVec = self.trainRes.lossVec[0]
            else:                lossVec = self.trainRes.lossVec[batch]
            uniform_input = self.fixData.uniform_input
            lossField = interpolate.LinearNDInterpolator(uniform_input, lossVec, fill_value=0.0)
            def lossFun(x, t=None):
                """Loss plot function handle."""
                if timeDependent:
                    Input = uf.hstack([x, t*np.ones([len(x),1])])
                else:
                    Input = x
                return lossField(Input)
        else: lossFun = None
        
        title = None                        # default title for plots
        
        if dim==1:
            # Plot snapshots of the exact and approximate solutions:
            if uf.isnone(cExact):
                Legend = []
                plt.figure(1)
                if pltFrmt=='.png': title = 'approximate solution'
                for t in tcoord:
                    contPlot.snap1Dt(cAppFun, t, figNum=1, title=title)
                    Legend.append('t={0:.2f}s'.format(t))
                plt.ylabel('solution')
                plt.legend(Legend)
                filename = 'cApp' + pltFrmt
                filepath = os.path.join(plotpath, filename)
                plt.savefig(filepath, dpi=300)
                plt.show()
                
            else:
                for t in tcoord:
                    plt.figure(1)
                    contPlot.snap1Dt(cExFun, t, figNum=1, lineOpt='b')
                    if pltFrmt=='.png': title = 't={0:.2f}s'.format(t)
                    contPlot.snap1Dt(cAppFun, t, figNum=1, lineOpt='r', title=title)
                    plt.ylabel('solution')
                    plt.legend(['exact solution', 'approximate solution'])
                    filename = 'cApp-' + 't={0:.2f}s'.format(t) + pltFrmt
                    filepath = os.path.join(plotpath, filename)
                    plt.savefig(filepath, dpi=300)
                    plt.show()
                
            # Plot snapshots of the solution error:
            if not uf.isnone(cExact):
                Legend = []
                plt.figure(1)
                if pltFrmt=='.png': title = 'solution error vs time'
                for t in tcoord:
                    contPlot.snap1Dt(cErrFun, t, figNum=1, title=title)
                    Legend.append('t={0:.2f}s'.format(t))
                plt.ylabel('error')
                plt.legend(Legend)
                filename = 'cErr' + pltFrmt
                filepath = os.path.join(plotpath, filename)
                plt.savefig(filepath, dpi=300)
                plt.show()

            # Plot snapshots of the residual:
            Legend = []
            plt.figure(1)
            if pltFrmt=='.png': title = 'residual vs time'
            for t in tcoord:
                contPlot.snap1Dt(resFun, t, figNum=1, title=title)
                Legend.append('t={0:.2f}s'.format(t))
            plt.ylabel('residual')
            plt.legend(Legend)
            filename = 'residual' + pltFrmt
            filepath = os.path.join(plotpath, filename)
            plt.savefig(filepath, dpi=300)
            plt.show()
            
            # Plot snapshots of the loss function for training points:
            if not uf.isnone(lossFun):
                Legend = []
                plt.figure(1)
                if pltFrmt=='.png': title = 'loss field vs time'
                for t in tcoord:
                    contPlot.snap1Dt(lossFun, t, figNum=1, title=title)
                    Legend.append('t={0:.2f}s'.format(t))
                plt.ylabel('loss')
                plt.legend(Legend)
                filename = 'lossField' + pltFrmt
                filepath = os.path.join(plotpath, filename)
                plt.savefig(filepath, dpi=300)
                plt.show()
                            
        
        elif dim==2:
            # Plot snapshots of the exact and approximate solutions and the residual:
            for t in tcoord:
                if pltFrmt=='.png': title = 'approximate solution - t={0:.2f}s'.format(t)
                cApp1 = contPlot.conPlot(cAppFun, t, title=title)
                filename = 'cApp-' + 't={0:.2f}s'.format(t) + pltFrmt
                filepath = os.path.join(plotpath, filename)
                plt.savefig(filepath, dpi=300)
                plt.show()
                
                if not uf.isnone(cExact):
                    if pltFrmt=='.png': title = 'exact solution - t={0:.2f}s'.format(t)
                    cEx1 = contPlot.conPlot(cExFun, t, title=title)
                    filename = 'cEx-' + 't={0:.2f}s'.format(t) + pltFrmt
                    filepath = os.path.join(plotpath, filename)
                    plt.savefig(filepath, dpi=300)
                    plt.show()
                    
                    if pltFrmt=='.png': title = 'error field - t={0:.2f}s'.format(t)
                    contPlot.conPlot(cErrFun, t, title=title)
                    filename = 'cErr-' + 't={0:.2f}s'.format(t) + pltFrmt
                    filepath = os.path.join(plotpath, filename)
                    plt.savefig(filepath, dpi=300)
                    plt.show()

                if pltFrmt=='.png': title = 'residual - t={0:.2f}s'.format(t)
                contPlot.conPlot(resFun, t, figNum=1, title=title)
                filename = 'res-' + 't={0:.2f}s'.format(t) + pltFrmt
                filepath = os.path.join(plotpath, filename)
                plt.savefig(filepath, dpi=300)
                plt.show()
                
                # Plot snapshots of the loss function for training points:
                if not uf.isnone(lossFun):
                    if pltFrmt=='.png': title = 'loss field - t={0:.2f}s'.format(t)
                    contPlot.conPlot(lossFun, t, title=title)
                    filename = 'lossField-' + 't={0:.2f}s'.format(t) + pltFrmt
                    filepath = os.path.join(plotpath, filename)
                    plt.savefig(filepath, dpi=300)
                    plt.show()
                               
                if not uf.isnone(cExact):
                    print('\napproximation error for t=%.2fs: %2.5f' % (t, uf.l2Err(cEx1, cApp1)) )
    


    def saveNNparam(self, dpOut=False, matOut=False, verbose=True, timeFirst=False):
        """
        Function to save the NN to numpy arrays and output them in .mat format 
        to be loaded to MATLAB upon request.
        We arrange the weight matrices such that the when multiplied with column
        input vector, they give the output. The biases are given in column 
        vector:                     o = W*i+b
        
        Inputs:
            dpOut: if True output matrices in diffPack readable '.m' format
            matOut: if True output matrices in MATLAB '.mat' format
            verbose: output text to screen if True
            timeFirst: if True adjust the weight matrix of the first layer
                so that temporal input is placed before spatial input
        
        output: list of lists that contain the weight and bias of each layer
        """
        # Data:
        tfData = self.tfData
        depth = tfData.depth
        sess = tfData.sess
        graph = tfData.graph
        with graph.as_default():
            trVar = tf.trainable_variables()    # load trainable variables
            
        PDE = self.PDE
        if not PDE.timeDependent: timeFirst = False
        else:                     dim = PDE.domain.dim
        
        # Path to store the variables:
        if dpOut or matOut:
            folderpath = self.trainRes.folderpath
            folderpath = os.path.join(folderpath, 'NN_parameters')
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)

        if len(trVar)%(2*(depth+1)):            # '1' accounts for last non-activated layer
            # (verytime the model is loaded new copies of the variables are added)
            print(len(trVar))
            print(depth)
            raise ValueError('number of weights and baises do not match the number of layers!')
        else:
            trVar = trVar[:2*(depth+1)]
        
        layers = []
        lnum = 0
        for i in np.arange(0,len(trVar),2):
            if verbose: print('Layer ' + str(lnum) + ':')
            lnum += 1
            layer = []
            
            if verbose: print('\tsaving weight matrix: ' + trVar[i].name)
            W = sess.run(trVar[i]).T
            if i==0 and timeFirst:              # switch columns corresponding to spatial and temporal inputs
                Wtmp = np.copy(W)
                W[:,0] = Wtmp[:,dim]
                W[:,1:dim+1] = Wtmp[:,:dim]
            layer.append(W)
            
            if verbose: print('\tsaving bias vector: ' + trVar[i+1].name + '\n')
            b = sess.run(trVar[i+1])[np.newaxis].T
            layer.append(b)
            layers.append(layer)
            
            if dpOut:
                    fieldname = 'W' + str(lnum)
                    filepath = os.path.join(folderpath, fieldname + '.m')
                    uf.mat2diffpack(filepath, fieldname, W)
                    
                    fieldname = 'B' + str(lnum)
                    filepath = os.path.join(folderpath, fieldname + '.m')
                    uf.mat2diffpack(filepath, fieldname, b)
            
            if matOut:
                filename = 'W' + str(lnum)
                filepath = os.path.join(folderpath, filename + '.mat')
                spio.savemat(filepath, {filename: W})
                filename = 'B' + str(lnum)
                filepath = os.path.join(folderpath, filename + '.mat')
                spio.savemat(filepath, {filename: b})
                
        return layers
        










