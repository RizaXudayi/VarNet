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
This file contains the classes to build a NN model using TensorFlow for the AD-PDE.

"""

#%% Modules:

import numpy as np
shape = np.shape
reshape = np.reshape
size = np.size

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.python.client import device_lib

from UtilityFunc import UF
uf = UF()

#%% Data Parallelism (Replicated Training)
        
class TFNN():
    """
    Class to construct the tensorflow computational graph and models accross devices.
    
    Attributes:
        dim
        inpDim
        seqLen
        depth
        layerWidth
        modelId
        activationFun
        timeDependent
        RNNdata
        graph: TensorFlow computational graph that stores the data
        model: NN model
        processorNum: number of processing units
        processors: processors to be used for training (GPU or CPU)
        controller: controller for parallel programming
        lossOpt [dict]: dictionary containing data to optimize the loss function
        learning_rate
        optimizer_name: to be used for training: Adam, RMSprop
        compTowers: list of 'NNModel' objects corresponding to the processors
        optimSetup variables:
            step
            saver
            loss
            optMinimize
            sess
    """
    
    def __init__(self, dim, inpDim, layerWidth, modelId, activationFun, timeDependent,
                 RNNdata, processors, controller, lossOpt, optimizer_name, learning_rate):
        """
        Class to construct the tensorflow variables, model, loss function on a 
        single computational graph accross multiple devices.
        
        Inputs:
            dim: dimension of the spatial domain
            inpDim: number of the NN inputs
            layerWidth [lNum x 1]: widths of the hidden layers
            modelId: indicator for the sequential TensorFlow model to be trained:
                'MLP': multi-layer perceptron with 'sigmoid' activation
                'RNN': recurrent network with 'gru' nodes
            activationFun: activation function used in layers
            timeDependent: boolean to specify time-dependent PDEs
            RNNdata: data of the RNN including the sequence length
            processors: processor(s) to be used for training (GPUs or CPU)
                data is split between processors if more than one is specified 
            controller (CPU or GPU): processor to contain the training data and
                perform optimization in the parallel setting
            lossOpt [dict]: dictionary containing data to optimize the loss function:
                integWflag: if True include integration weights in the loss function
                isSource: if True the source term is not zero in the PDE
            optimizer_name: to be used for training: Adam, RMSprop
            learning_rate: learning rate for Adam optimizer
        """
        # Error handling:
        depth = len(layerWidth)
        if type(activationFun)==str:
            activationFun = [activationFun]*depth
        elif type(activationFun)==list and len(activationFun)==1:
            activationFun = activationFun*depth
        elif not len(activationFun)==depth:
            raise ValueError('activation function list is incompatible with number of layers!')
            
        processors, flag = self.get_processor(processors)
        puNum = len(processors)
        for i in range(puNum):
            if not flag[i]:
                raise ValueError('requested processor %i is unavailable!' % i)
        controller, flag = self.get_processor(controller)
        controller = controller[0]
        if not flag:
            raise ValueError('requested controller is unavailable!')
        
        if learning_rate<0.0: raise ValueError('learning rate must be positive!')
        
        if optimizer_name.lower()=='rms': optimizer_name = 'rmsprop'
        if not (optimizer_name.lower()=='adam' or optimizer_name.lower()=='rmsprop'):
            raise ValueError('unknown optimizer requested!')
        
        # Processor assignment:
        gpuList = self.get_available_gpus()
        if uf.isnone(processors) and uf.isnone(controller):
            if uf.isempty(gpuList):
                print('\nUsing CPU for training ...')
                processors = ['/device:CPU:0']
                controller = '/device:CPU:0'
            else:
                print('\nUsing first available GPU for training ...')
                processors = gpuList[0:1]
                controller = gpuList[0]
        elif not uf.isnone(processors) and uf.isnone(controller):
            if type(processors)==list and len(processors)==1:
                controller = processors[0]
            elif type(processors)==str:
                controller = processors
            else:
                print('\nUsing the CPU as the controller for parallel training ...')
                controller = 'CPU:0'
        elif uf.isnone(processors) and not uf.isnone(controller):
            if len(gpuList)<=1:
                raise ValueError('not enough GPUs for parallel training!')
            else:
                print('\nUsing all available GPUs for parallel training ...')
                processors = gpuList
        
        # Save data:
        self.dim = dim
        self.inpDim = inpDim
        self.depth = depth
        self.layerWidth = layerWidth
        self.modelId = modelId
        self.activationFun = activationFun
        self.timeDependent = timeDependent
        self.RNNdata = RNNdata
        self.processorNum = puNum           # number of processing units
        self.processors = processors
        self.controller = controller
        self.lossOpt = lossOpt
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        
        # Create the computational graph and central optimizer:
        graph = tf.Graph()                  # computational graph of the class
        with graph.as_default(), tf.device(controller):
            self.defModel()                 # define the shared model
            
            if optimizer_name.lower()=='adam':
                optimizer = tf.train.AdamOptimizer(learning_rate, name='Optimizer')
            elif optimizer_name.lower()=='rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate, name='Optimizer')
        
        self.graph = graph
        self.optimizer = optimizer
        self.towerSetup()
        self.optimSetup()                   # setup the optimization nodes and start the session
        
        

    def defModel(self):
        """Function to setup the NN model according to PDE and MOR parameters."""
        
        if hasattr(self, 'model'):
            raise Exception('model is already defined!')
                
        inpDim = self.inpDim
        layerWidth = self.layerWidth
        depth = self.depth
        modelId = self.modelId
        activFun = self.activationFun
        
        with tf.name_scope('defModel'):
            model = Sequential()
            if modelId == 'MLP':
                model.add(layers.Dense(layerWidth[0],
                                       activation = activFun[0],
                                       input_shape = (inpDim,),
                                       kernel_initializer = 'glorot_uniform',
                                       name='dense_0'))
                for d in range(depth-1):                    # loop over network depth
                    name = 'dense_' + str(d+1)
                    model.add(layers.Dense(layerWidth[d+1],
                                       activation=activFun[d+1],
                                       kernel_initializer = 'glorot_uniform',
                                       name=name))
                    
            elif modelId == 'RNN':
                seqLen = self.RNNdata.seqLen                # number of time discretizations
                model.add(layers.GRU(layerWidth[0],
                                     activation = activFun[d+1],
                                     recurrent_activation = 'hard_sigmoid',
                                     input_shape = (seqLen, inpDim),
                                     return_sequences = True,
                                     kernel_initializer = 'he_uniform',
                                     unroll=True,
                                     name='gru_0'))
                for d in range(depth-1):                    # loop over network depth
                    name = 'gru_' + str(d+1)
                    model.add(layers.GRU(layerWidth[d+1],
                                     activation = activFun,
                                     recurrent_activation = 'hard_sigmoid',
                                     return_sequences = True,
                                     kernel_initializer = 'he_uniform',
                                     unroll=True,
                                     name=name))
        
            model.add(layers.Dense(1, name='output'))      # output layer
            
        # Output model information:
        print('\nNumber of inputs:', inpDim)
        model.summary()
        
        # Store data:
        self.model = model                                  # add the model to the attributes
        
    
    
    def towerSetup(self):
        """
        Function to setup the computational towers located on the requested
        """
        
        if hasattr(self, 'compTowers'):
            raise Exception('computational towers are already defined!')
            
        # Load data:
        dim = self.dim
        inpDim = self.inpDim
        modelId = self.modelId
        timeDependent = self.timeDependent
        RNNdata = self.RNNdata
        puNum = self.processorNum           # number of processing units
        processors = self.processors
        lossOpt = self.lossOpt
        graph = self.graph
        model = self.model
        optimizer = self.optimizer
        
        # Create the computational graph and central optimizer:
        compTowers = []                     # instnaces of 'NNModel' for computational towers (processors)
        with graph.as_default():
            for i, pu in enumerate(processors):      # loop over processors
                name = 'tower_{}'.format(i)
                
                with tf.device(pu), tf.name_scope(name):
                    if puNum>1: print('\n\nTower %i:' % i)
                    
                    # Create an instance of class 'NNModel' and add it to list:
                    compTowers.append(NNModel(dim, inpDim, modelId, timeDependent,
                                              RNNdata, lossOpt, model, optimizer))
                    if puNum>1: print('\n' + '-'*80)
        
        # Save data:
        self.compTowers = compTowers        # computational towers to be used for training
        


    def optimSetup(self):
        """Function to define the saver and optimization nodes. """
        
        if hasattr(self, 'optMinimize'):
            raise Exception('optimization node is already defined!')
            
        # Load data:
        puNum = self.processorNum
        controller = self.controller
        graph = self.graph
        optimizer = self.optimizer
        compTowers = self.compTowers
        
        with graph.as_default():
            saver = tf.train.Saver(max_to_keep=2)           # save optimization state
            print('\nsaver constructed.')
            
            with tf.name_scope('optimSetup'), tf.device(controller):
                grad = self.sum_grads()
                step = tf.train.get_or_create_global_step()
                apply_grad = optimizer.apply_gradients(grad, global_step=step, name='optimizer')
                
                loss = tf.reduce_sum([compTowers[pu].loss for pu in range(puNum)], name='loss')
                BCloss = tf.reduce_sum([compTowers[pu].BCloss for pu in range(puNum)], name='BCloss')
                ICloss = tf.reduce_sum([compTowers[pu].ICloss for pu in range(puNum)], name='ICloss')
                varLoss = tf.reduce_sum([compTowers[pu].varLoss for pu in range(puNum)], name='varLoss')
                lossVec = tf.concat([compTowers[pu].lossVec for pu in range(puNum)], axis=0, name='lossVec')
                
                print('optimizer constructed.\n')
                
            config = tf.ConfigProto(log_device_placement=False)
#            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)                # start a new training session
            sess.run(tf.global_variables_initializer())     # initialize all variables
        
        # Store data:
        self.step = step
        self.saver = saver
        self.optMinimize = apply_grad
        self.sess = sess
        
        self.loss = loss
        self.BCloss = BCloss                                # BCs loss component
        self.ICloss = ICloss                                # ICs loss component
        self.varLoss = varLoss                              # variational loss component
        self.lossVec = lossVec                              # variational loss field



    def sum_grads(self):
        """
        Function to compute the gradient of shared trainable variable across all towers.
        Note that this function provides a synchronization point across all towers.
        The output of the 'compute_gradients()', used in 'NNModel', is a list of 
        (gradient, variable) tuples that ranges over the trainbale variables.
                
        Output: list of pairs of (gradient, variable)
        """
        # Load data:
        puNum = self.processorNum
        compTowers = self.compTowers
        
        if puNum==1:
            return compTowers[0].grad
        
        # Collect gradients from different towers:
        tower_grads = []
        for tower in compTowers:
            tower_grads.append(tower.grad)
            
        # Sum gradients for individual variables:
        grad_varTot = []
        for i, grad_var in enumerate(zip(*tower_grads)):        # loop over trainable variables

            # Each grad_and_vars looks like the following:
            # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = [g for g, _ in grad_var]
            grad = tf.reduce_sum(grads, 0, name = 'grad_sum_'+str(i))
    
            # Since variables are are shared among towers, we use the first tower's variables:
            var = grad_var[0][1]
            grad_var = (grad, var)
            grad_varTot.append(grad_var)
        
        return grad_varTot
            


    def get_available_gpus(self):
        """
        Function to return the list of all visible GPUs.
        """
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']


    def get_processor(self, processors):
        """
        Function to determine if requested processors (CPU or GPU) is available.
        The processors should be specified as 'CPU:i' or 'GPU:i' where i is the
        index of the processor.
        """
        if not type(processors)==list:
            processors = [processors]
        puNum = len(processors)                         # number of processing units
        if uf.isnone(processors):
            flag = [True]*puNum
            return processors, flag
        
        local_device_protos = device_lib.list_local_devices()
        flag = [False]*puNum
        for i in range(puNum):
            for x in local_device_protos:
                if x.name[8:].lower()==processors[i].lower():
                    flag[i] = True
                    break
            processors[i] = '/device:' + processors[i].upper()
        return processors, flag
    
    
        
    def assign_to_device(self, processor, controller):
        """
        Returns a function to place variables on the controller.
    
        Inputs:
            processor: device for everything but variables
            controller: device to put the trainable variables on
    
        If 'controller' is not set, the variables will be placed on the default processor.
        The best processor for shared varibles depends on the platform as well 
        as the model. Start with 'CPU:0' and then test 'GPU:0' to see if there 
        is an improvement.
        """
        varNames = ['Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
                  'MutableHashTableOfTensors', 'MutableDenseHashTable']
        
        def _assign(op):
            node_def = op if isinstance(op, tf.NodeDef) else op.node_def
            if node_def.op in varNames:
                return controller
            else:
                return processor
        return _assign



#%% TensorFlow Model Class:

class NNModel():
    """
    Class to construct the tensorflow computational model on a given device.
    
    Attributes:
        dim
        inpDim
        seqLen
        modelId
        model: NN model
        modelGrad variables:
            Input
            dM_dt
            dM_dx
            d2M_dx2
        LossFun variables:
            bDof
            w
            intShape
            detJ
            integW
            biDimVal
            source
            gcoef
            N
            dNt
            biInput
            biLabel
            loss
            detJvec
        grad: gradient of the loss function wrt to trainable variables
        Residual variables:
            diff
            vel
            diff_dx
            res
    """
    
    def __init__(self, dim, inpDim, modelId, timeDependent, RNNdata, lossOpt,
                 model, optimizer):
        """
        Class to construct the tensorflow variables, model, loss function on a 
        single computational graph.
        
        Input:
            dim: dimension of the spatial domain
            inpDim: number of the NN inputs
            modelId: indicator for the sequential TensorFlow model to be trained:
                'MLP': multi-layer perceptron with 'sigmoid' activation
                'RNN': recurrent network with 'gru' nodes
            timeDependent: boolean to specify time-dependent PDEs
            RNNdata: data of the RNN including the sequence length
            lossOpt [dict]: dictionary containing data to optimize the loss function:
                integWflag: if True include integration weights in the loss function
                isSource: if True the source term is not zero in the PDE
            model: NN model that is shared among the computational towers
            optimizer: optimization node to update the weights
        """
        # Save data:
        self.dim = dim
        self.inpDim = inpDim
        self.modelId = modelId
        self.RNNdata = RNNdata
        self.timeDependent = timeDependent
        self.optimizer = optimizer
        self.model = model                  # shared model among computational towers
        self.modelGrad()                    # model gradients
        self.LossFun(lossOpt)               # define the loss function
        self.computeGrad()                  # compute the gradient of loss function
        self.Residual()                     # function to compute the PDE residual field
    
    
    
    def modelGrad(self):
        """
        Function to define the model gradients. We assume that the order of 
        inputs to the NN are 'x, t, MOR parameters'.
        """
        # Error handling:
        if hasattr(self, 'Input'):
            raise ValueError('Variables and gradients are already defined!')
        
        # Data:
        dim = self.dim
        model = self.model
        modelId = self.modelId
        inpDim = self.inpDim
        
        with tf.name_scope('model_grad'):
            Input = tf.placeholder(tf.float32, name='Input')    # inner-domain nodes
            
            if modelId == 'MLP':
                Input.set_shape([None, inpDim])
                
                dM_dx = tf.gradients(model(Input), Input)[0]    # first order derivative wrt inputs
                if self.timeDependent:
                    dM_dt = dM_dx[:,dim:dim+1]
                else:
                    dM_dt = None
                dM_dx = dM_dx[:,0:dim]                          # keep only relevant derivatives
                
                d2M_dx2 = tf.gradients(dM_dx[:,0], Input)[0][:,0:1]
                for d in range(1,dim):                          # second order derivative wrt x_d
                    d2M_dx2 = d2M_dx2 + tf.gradients(dM_dx[:,d], Input)[0][:,d:(d+1)]
                
            elif modelId == 'RNN':
                seqLen = self.RNNdata.seqLen                    # number of time discretizations
                Input.set_shape([None, seqLen, inpDim])
    
                dM_dx = self.gradRNN(model, Input)              # first order derivative wrt inputs
                dM_dt = dM_dx[:,:,dim:dim+1]                    # derivative wrt to time
                dM_dx = dM_dx[:,:,0:dim]                        # keep only relevant derivatives
                
                d2M_dx2 = None                                  # it is computationally very demanding to compute
#                d2M_dx2 = tf.gradients(dM_dx[:,:,0], Input)[0][:,:,0:1]
#                for d in range(1,dim):                          # second order derivative wrt x_d
#                    d2M_dx2 = d2M_dx2 + tf.gradients(dM_dx[:,:,d], Input)[0][:,:,d:(d+1)]
                
        # Store data:
        self.Input = Input
        self.dM_dt = dM_dt
        self.dM_dx = dM_dx
        self.d2M_dx2 = d2M_dx2
        
        
    def LossFun(self, lossOpt):
        """
        Define trainig loss function.
        
        Note: the determinant of the Jacobian matrix is not properly applied.
              Specifically, detJ**2 must multiply the numerical integration
              result and not detJ. This is because we sqaure the variational
              term below. This is not a problem for identical test functions 
              since detJ acts as a scaling and can be combined with weight value,
              but for non-identical test function supports, the calculation is
              inaccurate.
        
        Inputs:
            lossOpt [dict]: dictionary containing data to optimize the loss function:
                integWflag: if True include integration weights in the loss function
                isSource: if True the source term is not zero in the PDE
        """
        # Error handling:
        if hasattr(self, 'loss'):
            raise ValueError('Loss function is already defined!')
        
        print('\nloss function in construction ...')
        
        # Data:
        dim = self.dim
        inpDim = self.inpDim
        timeDependent = self.timeDependent
        modelId = self.modelId
        model = self.model
        Input = self.Input
        dM_dx = self.dM_dx
        
        # Define the loss computation node:
        with tf.name_scope('loss_fun'):
            # Define the variables:
            biInput = tf.placeholder(tf.float32, name='biInput')    # boundary nodes
            biInput.set_shape([None, inpDim])
            biLabel = tf.placeholder(tf.float32, name='biLabel')    # boundary labels
            biLabel.set_shape([None, 1])
            bDof = tf.placeholder(tf.int32, name='bDof')            # total number of boundary nodes over space-time
            w = tf.placeholder(tf.float32, name='w')                # weights for loss function
            intShape = tf.placeholder(tf.int32, name='intShape')    # number of integration points per element
            detJ = tf.placeholder(tf.float32, name='detJ')          # determinant of the Jacobian
            integW = tf.placeholder(tf.float32, name='integW')      # integration weights
            biDimVal = tf.placeholder(tf.float32, name='biDimVal')  # dimensional correction for boundary-initial condition
            source = tf.placeholder(tf.float32, name='source')      # source term
            source.set_shape([None, 1])
            gcoef = tf.placeholder(tf.float32, name='gcoef')        # coefficient of \nabla c
            gcoef.set_shape([None, dim])
            N = tf.placeholder(tf.float32, name='N')                # FE basis function values at integration points
            N.set_shape([None, 1])
            dNt = tf.placeholder(tf.float32, name='dNt')            # time-derivative of the FE basis functions at integration points
            dNt.set_shape([None, 1])
            detJvec = tf.placeholder(tf.bool, shape=[], name='detJvec')
    
            # Compute the PDE components:
            if modelId == 'MLP':
                biVal = model(biInput)
                Val = model(Input)
                grad = dM_dx
                
            elif modelId == 'RNN':
                RNNdata = self.RNNdata
                seqLen = RNNdata.seqLen                             # number of time discretizations
                bdof = RNNdata.bdof                                 # number of boundary nodes
                integInd = RNNdata.integInd                         # mapping from RNN to numerical integration
                
                Val = model(Input)
                bVal = tf.reshape(Val[0:bdof,:,0], [bdof*seqLen,1])
                iVal = Val[bdof:,0:1,0]
                biVal = tf.concat([bVal, iVal], axis=0)
                
                Val = tf.gather_nd(Val, indices=integInd, name="gatherNd")
                grad = tf.gather_nd(dM_dx, indices=integInd, name="gatherNd")
            
            # Boundary-initial conditions:
            biCs = biDimVal*(biVal - biLabel)**2
            bCs = biCs[:bDof,0:1]                                   # boundary condition error term
            bCs = tf.reduce_mean(bCs)
            if timeDependent:
                iCs = biCs[bDof:,0:1]                               # initial condition error term
                iCs = tf.reduce_mean(iCs)                           # returns nan for empty tensor
            else:
                iCs = tf.constant(0.0, dtype=tf.float32, name='constIC')
            
            # Gauss-Legender integration of the PDE '\nabla c (kapa*\nabla v + bbu*v) - c*vdot':                    
            int1 = tf.multiply(grad, gcoef)                         # \nabla c contribution to integrand
            int1 = tf.reduce_sum(int1, axis=-1, keepdims=True)      # inner-product (sum over coordinates)
            if timeDependent: int1 = int1 - tf.multiply(Val, dNt)   # contribution of time-derivative to integrand
            if lossOpt['isSource']:
                int1 = int1 - tf.multiply(source, N)                # contribution of source-term to integrand
            
            int1 = tf.reshape(int1, intShape)                       # reshapbe back for each training point
            if lossOpt['integWflag']: int1 = integW*int1            # integration weights
            int1 = tf.reduce_sum(int1, axis=-1, keepdims=True)**2   # sum over integration points and elements
            int2 = tf.cond(detJvec,
                           lambda: tf.reduce_sum(detJ*int1),        # sum of errors at training points
                           lambda: detJ*tf.reduce_sum(int1) )       # move 'detJ' outside for computational efficiency
                
            loss = w[0]*bCs + w[1]*iCs + w[2]*int2                  # loss value used for training
            
            lossVec = detJ*int1                                     # vector of loss values across domain
        
        # Store data:
        self.bDof = bDof
        self.w = w
        self.intShape = intShape
        self.detJ = detJ
        self.integW = integW
        self.biDimVal = biDimVal
        self.source = source
        self.gcoef = gcoef
        self.N = N
        self.dNt = dNt
        self.biInput = biInput
        self.biLabel = biLabel
        self.loss = loss
        self.detJvec = detJvec
        
        self.BCloss = bCs                                           # BCs loss component
        self.ICloss = iCs                                           # ICs loss component
        self.varLoss = int2                                         # variational loss component
        self.lossVec = lossVec                                      # variational loss field
        
        print('loss function constructed.\n')
            
        
        
    def computeGrad(self):
        """
        Function to compute the gradient of the loss function wrt the trainable variables.
        """
        if hasattr(self, 'grad'):
            raise Exception('grad node is already defined!')
        
        print('gradient node in construction ...')
        
        # Data:
        loss = self.loss
        optimizer = self.optimizer
        
        with tf.name_scope('compute_grad'):
            grad = optimizer.compute_gradients(loss)        # returns a list of (gradient, variable) pairs
            
        # Store data:
        self.grad = grad
        
        print('gradient node constructed.\n')



    def Residual(self):
        """Function to compute the residual of the PDE."""
        
        if hasattr(self, 'residual'):
            raise Exception('PDE-residual is already defined!')
        
        print('residual function in construction ...')
        
        # Data:
        dim = self.dim
        modelId = self.modelId
        dM_dt = self.dM_dt
        dM_dx = self.dM_dx
        d2M_dx2 = self.d2M_dx2
        source = self.source
        
        # Tensorflow variables:
        with tf.name_scope('residual'):
            diff = tf.placeholder(tf.float32, name='diff')              # diffusivity values at integration points
            diff.set_shape([None, 1])
            vel = tf.placeholder(tf.float32, name='vel')                # velocity vector field values at integration points
            vel.set_shape([None, dim])
            diff_dx = tf.placeholder(tf.float32, name='diff_dx')        # gradient of the diffusivity field at integration points
            diff_dx.set_shape([None, dim])
            
            # Compute the gradient and Hessian:
            if modelId=='MLP':
                gradt = dM_dt
                grad = dM_dx
                hess = d2M_dx2
                
                # Define the residual computation node:
                if gradt is None: res = 0
                else:     res = - gradt
                res = res + tf.multiply(diff, hess)
                res = res - tf.reduce_sum( tf.multiply(vel-diff_dx, grad), axis=-1, keepdims=True )
                res = res + source
                
            elif modelId=='RNN':
                seqLen = self.RNNdata.seqLen
                dof = tf.shape(dM_dx)[0]
    #                gradt = tf.reshape(dM_dt, [dof*seqLen, 1])
    #                grad = tf.reshape(dM_dx, [dof*seqLen, dim])
    #                hess = tf.reshape(d2M_dx2, [dof*seqLen, 1])
                
                res = tf.zeros([dof*seqLen, 1])
                # Residual for RNN is not computationally feasible.
            
        # Store data:
        self.diff = diff
        self.vel = vel
        self.diff_dx = diff_dx
        self.residual = res
        
        print('residual function constructed.\n')



################################# deprecated ##################################
        
    def optimSetup(self):
        """Function to define the optimization nodes. """
        
        if hasattr(self, 'optimizer'):
            raise Exception('optimization node is already defined!')
        
        print('optimization node in construction ...')
        
        # Data:
        graph = self.graph
        processor = self.processor
        loss = self.loss
        
        with graph.as_default(), tf.name_scope("compute_gradients"):
            saver = tf.train.Saver(max_to_keep=2)                               # save optimization state
            print('saver constructed.\n')
            with tf.device(processor):
                step = tf.train.get_or_create_global_step()                     # global optimization step
                optimizer = tf.train.AdamOptimizer(name='Optimizer')
                optMinimize = optimizer.minimize(loss, global_step=step)        # add loss function
                resetList = [optimizer.get_slot(var, name) for var in tf.trainable_variables() for name in optimizer.get_slot_names()]
                resetOpt = tf.variables_initializer(resetList)                  # reset node for slot variables (history) of the optimizer for trainable variables
            print('optimizer constructed.')
            
            config = tf.ConfigProto(log_device_placement=False)
#            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)                                    # start a new training session
        
        # Store data:
        self.step = step
        self.saver = saver
        self.optimizer = optimizer
        self.optMinimize = optMinimize
        self.resetOpt = resetOpt
        self.sess = sess
    
    
    
    def gradRNN(self, model, Input):
        """
        Function to compute the gradient of the RNN wrt its inputs. Note that 
        for RNNs, each output depends on all inputs before it. If we assume that
        the spatial input is constant for each batch, then the total derivative
        is simply the sum of derivatives of each output wrt all inputs up to that
        output. See the notes for details.
        
        Inputs:
            model: RNN model
            Input: Input to the model
        """
        seqLen = self.RNNdata.seqLen
        val = model(Input)                                          # pass the input to the model
        
        # Compute gradient for each output wrt all inputs and sum them up:
        grad = []
        print()
        for t in range(seqLen):
            print('calculating the gradient for output number ', t+1)
            gradTmp = tf.gradients(val[:,t,:], Input)[0]            # gradient for each output
            grad.append(gradTmp)
        grad = tf.stack(grad, axis=1)                               # stack all gradients into a tensor
        grad = tf.reduce_sum(grad, axis=2, keepdims=False)          # sum over Input for each output
        
        return grad






        
        










