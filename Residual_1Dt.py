# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:37:29 2018

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
This code segment solves the 1Dt AD-PDE using the residual loss function to
highlight the advantages of the variational form.

An animation of the final solution corresponding to the trained NN can be found
here: https://vimeo.com/350138714

The corresponding benchmark problem can be found in:
Abdelkader Mojtabi and Michel O Deville. One-dimensional linear Advection-Diffusion equation:
Analytical and Finite Element solutions. Computers & Fluids, 107:189–195, 2015.

"""

#%% Modules:

import os

import tensorflow as tf

import numpy as np
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
sh = np.shape
reshape = np.reshape
import numpy.linalg as la
import time

import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
sess = tf.Session()

#%% 1D AD-PDE: closed-form solution

u = 1.0                         # velocity magnitude
D = 0.1/pi                      # diffusivity
T = 2.0                         # maximum time bound
l1 = -1;        l2 = 1          # domain bounds

# BCs (values):
c0 = 0.0;       c1 = 0.0

# Initial condition:
def IC(x):
    return -sin(pi*x)

def cExact(t,x, trunc = 800):
    # Function to compute the analytical solution as a Fourier series expansion.
    # Inputs:
    #       t: column vector of times
    #       x: row vector of locations
    #       trunc: truncation number of Fourier bases
    
    # Adjust the shape of variables:
    p = np.arange(0, trunc+1.0)
    p = reshape(p, [1, 1, trunc+1])
    t_disc_num = len(t)
    t = reshape(t, [t_disc_num, 1, 1])
    x_disc_num = len(x)
    x = reshape(x, [1, x_disc_num, 1])
    
    cT = 16*pi**2*D**3*u*exp(u/D/2*(x-u*t/2))                           # time solution
    
    cX1_n = (-1)**p*2*p*sin(p*pi*x)*exp(-D*p**2*pi**2*t)                # numerator of first space solution
    cX1_d = u**4 + 8*(u*pi*D)**2*(p**2+1) + 16*(pi*D)**4*(p**2-1)**2    # denominator of first space solution
    cX1 = np.sinh(u/D/2)*np.sum(cX1_n/cX1_d, axis=-1, keepdims=True)    # first component of spacial solution
    
    cX2_n = (-1)**p*(2*p+1)*cos((p+0.5)*pi*x)*exp(-D*(2*p+1)**2*pi**2*t/4)
    cX2_d = u**4 + (u*pi*D)**2*(8*p**2+8*p+10) + (pi*D)**4*(4*p**2+4*p-3)**2
    cX2 = np.cosh(u/D/2)*np.sum(cX2_n/cX2_d, axis=-1, keepdims=True)    # second component of spacial solution
    
    return np.squeeze(cT*(cX1+cX2))

# Plot the exact solution:
time2 = np.arange(0.0+0.1, T, 0.2)
coord2 = np.arange(l1,l2+0.01,0.01)

# Analytical solution (see the reference for sanity check):
cEx = cExact(time2, coord2)
cInit = reshape(IC(coord2), [1, len(coord2)])
cEx = np.concatenate([cInit, cEx], axis=0)
time2 = np.append(0, time2)

print('\nPeclet number: %2.2f' % (u*(l2-l1)/D))

plt.figure()
Legend = []
for t in range(len(time2)):
    pltH = plt.plot(coord2, cEx[t, :])
    Legend.append('t={0:.2f}s'.format(time2[t]))
plt.legend(Legend)
plt.xlabel('x');    plt.ylabel('c(t,x)')
plt.grid(True)
plt.show()


#%% Define the neural network model:

model = Sequential()
model.add(layers.Dense(20, activation='sigmoid', input_shape=(2,), kernel_initializer = 'glorot_uniform'))
#model.add(layers.Dense(20, activation='sigmoid', kernel_initializer = 'glorot_uniform'))
model.add(layers.Dense(1))

model.summary()

#%% Derivatives:

# Variables used in the code:
x = tf.placeholder(tf.float32)
x.set_shape([None, 2])
xi = tf.placeholder(tf.float32)                 # initial condition nodes
xi.set_shape([None, 2])
xb = tf.placeholder(tf.float32)                 # boundary condition nodes
xb.set_shape([None, 2])

dM_dx = tf.gradients(model(x), x)[0]            # first order derivative wrt time and space
d2M_dx2 = tf.gradients(dM_dx[:,1], x)[0]        # second order derivative
d2M_dx2 = d2M_dx2[:,1]                          # second order derivative wrt space

#%% Training Points:

nt = 1200                                       # number of temporal training points
ht = T/nt                                       # element size
Time = np.arange(0+ht, T, ht)
Time = np.reshape(Time, [nt-1, 1])

ns = 80                                         # number of spatial training points
hs = (l2-l1)/ns                                 # element size
coord = np.arange(l1+hs, l2, hs)                # spatial training points
coord = np.reshape(coord, [ns-1, 1])

# Input to model:
Coord = np.reshape(coord, [ns-1, 1])            # reshape input so that model can evaluate it
Coord2 = np.tile(Coord, [nt-1, 1])              # repeat spatial training points for each time instace
Time2 = np.repeat(Time, repeats=ns-1, axis=0)   # time instances corresponding to spatial training points
Input = np.concatenate([Time2, Coord2], axis=1)

# Initial condition input:
iInput = np.concatenate([np.zeros([ns-1, 1]), coord], axis=1)
ci = IC(coord)

# Boundary condition input:
Time2 = reshape(np.append([0], Time), [nt, 1])
bInput1 = np.concatenate([Time2, l1*np.ones([nt, 1])], axis=1)
bInput2 = np.concatenate([Time2, l2*np.ones([nt, 1])], axis=1)
bInput = np.concatenate([bInput1, bInput2], axis=0)
cb = np.concatenate([c0*np.ones([nt, 1]), c1*np.ones([nt, 1])])

#%% Loss function:

def loss(w=[1.e7, 1.e7, 1.e5]):
    
    # Initial conditions:
    ICs = tf.reduce_sum((model(xi) - ci)**2)
    
    # Boundary conditions:
    BCs = tf.reduce_sum((model(xb) - cb)**2)
    
    # PDE residual:
    res = tf.reduce_sum((dM_dx[:,0] + u*dM_dx[:,1] - D*d2M_dx2)**2)

    return w[0]*ICs + w[1]*BCs + w[2]*res


#%% Learning Rate:

global_step = tf.train.get_or_create_global_step()          # global optimization step

# Constant learning rate:
learning_rate = 1.e-3

# Inverse time decay:
#learning_rate = 1.e-3
#decay_steps = 1000.0
#decay_rate = 0.4
#learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False)

# Piece-wise constant learning rate:
#boundaries = [100, 2000, 5000, 10000]
#values = [1e-3, 5e-4, 1e-3, 5e-3, 1e-2]
#learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

#%% Optimizer:

optimizer = tf.train.AdamOptimizer()
#optimizer = tf.train.RMSPropOptimizer(learning_rate)

# This line is CRUCIAL in avoiding addition of new nodes and slowing down the training!!!
Loss = loss()
optimizer = optimizer.minimize(Loss,                        # add loss function
                               global_step=global_step)

saver = tf.train.Saver(max_to_keep=1)                       # save optimization state

sess.run(tf.global_variables_initializer())                 # initialize all variables

#%% Folder to Store Checkpoints:

#filepath = '/Users/Riza/Documents/Python/TF_checkpoints'
filepath = '/home/reza/Documents/Python/TF_checkpoints'           # Linux
for the_file in os.listdir(filepath):
    file_path = os.path.join(filepath, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

#%% Training initialization:
        
train_loss = []
min_loss = float('inf')
epoch_time = []
filepath2 = os.path.join(filepath, 'best_model')

#%% Training:

epochs = range(500000)

for epoch in epochs:                                        # loops over all training data every time
    t = time.clock()
    current_loss, _ = sess.run([Loss, optimizer], {x: Input, xi: iInput, xb: bInput})
    
    if epoch%100==99 and (min_loss-current_loss)>1.e-3:
        min_loss = current_loss
        saver.save(sess, filepath2, global_step=global_step) # best model so far
        print('  best model loss: %2.5f' % min_loss)
        
#    if epoch>1000 and np.abs(np.sum(np.diff(train_loss[-10:]))/(train_loss[-1]+1.e-3))<1.e-8:
    if current_loss<1.e-8:
        print('the iterations converged!')
        break
    
    if epoch<100 or (epoch<500 and epoch%10==0) or (epoch<1500 and epoch%100==0) or (epoch%500==0):
        print('Epoch %2d: loss = %2.5f' % (epoch, current_loss))
        train_loss.append(current_loss)
        epoch_time.append(time.clock()-t)
        
    if epoch%1000==999:
        plt.figure()
        plt.semilogy(train_loss)
        plt.grid(True)
        plt.ylabel('loss function');    plt.xlabel('epochs')
        plt.show()


print('average epoch time: {:.3}sec'.format(np.mean(epoch_time)))

plt.figure()
plt.plot(epoch_time)
plt.grid(True)
plt.ylabel('time per epoch (s)');    plt.xlabel('epochs')

#%% Convergence Plot:

pltEpoch = 100          # number of the initial data points to discard

plt.figure()
plt.semilogy(train_loss[pltEpoch:])
plt.grid(True)
plt.ylabel('loss function');    plt.xlabel('epochs')

#%% Final cost:

# Load the best stored model:
saver.restore(sess, saver.last_checkpoints[-1])

print('IC error:', sess.run(loss([1., 0., 0.]), {x: Input, xi:iInput, xb: bInput}))
print('BC value:', sess.run(loss([0., 1., 0.]), {x: Input, xi:iInput, xb: bInput}))
print('integral value:', sess.run(loss([0., 0., 1.]), {x: Input, xi:iInput, xb: bInput}))
print('total objective value:', sess.run(loss([1., 1., 1.]), {x: Input, xi:iInput, xb: bInput}))
print('total objective value:', sess.run(loss(), {x: Input, xi:iInput, xb: bInput}), '(w adjusted)')

#%% Plot true and predited solutions:

Coord2 = np.transpose(np.tile(coord2, reps=[1, len(time2)]))
Time2 = reshape(np.repeat(time2, repeats=len(coord2)), [len(Coord2), 1])
input2 = np.concatenate([Time2, Coord2], axis=1)
cApp = sess.run(model(x), {x:input2})
cApp = reshape(cApp, [len(time2), len(coord2)])

appErr = []
plt.figure()
for t in range(len(time2)):
    plt.plot(coord2, cEx[t,:], 'b')
    plt.xlabel('x')
    plt.ylabel('concentration')
    plt.grid(True)
    plt.plot(coord2, cApp[t,:], 'r')
    plt.title('t={0:.2f}s'.format(time2[t]))
    plt.show()
    appErr.append(la.norm(cApp[t,:]-cEx[t,:])/la.norm(cEx[t,:]))

plt.figure()
plt.plot(time2, appErr)
plt.xlabel('t')
plt.ylabel('estimation error')
plt.grid(True)
plt.show()

print('average normalized concentration error: {:.3}'.format(np.mean(appErr)))

#%% Equation Check:

grad = sess.run(dM_dx, {x: input2})
hess = sess.run(d2M_dx2, {x: input2})
res = grad[:,0] + u*grad[:,1] - D*hess
res = reshape(res, [len(time2), len(coord2)])

resErr = []
Legend = []
plt.figure()
for t in np.arange(1,len(time2)):
    plt.plot(coord2, res[t,:])
    Legend.append('t={0:.2f}s'.format(time2[t]))
    resErr.append(la.norm(res[t,:]))
plt.xlabel('x')
plt.ylabel('residual')
plt.legend(Legend)
plt.grid(True)
plt.show()

print('average residual error: {:.3}'.format(np.mean(resErr)))


###############################################################################
#%% Load the model when the folder path has changed:

# Load the best model from folder:
#filepath = "/home/reza/Documents/Python/TF_checkpoints"
#new_path = "/Users/Riza/Documents/Python/TF_checkpoints"
#check_path = os.path.join(new_path, 'checkpoint')
#
#with open(check_path, 'r') as myfile:
#    data = myfile.read()
#
#with open(check_path, 'w') as myfile:
#    new_data = data.replace(filepath, new_path)
#    myfile.write(new_data)
#
#filepath3 = os.path.join(new_path, 'best_model-500000.meta')
#saver = tf.train.import_meta_graph(filepath3)
#saver.restore(sess, tf.train.latest_checkpoint(new_path))

#%% Animation:

#time2 = np.linspace(0.0, T, 101)
#
## Analytical solution:
#cEx = cExact(time2[1:], coord2)
#cInit = reshape(IC(coord2), [1, len(coord2)])
#cEx = np.concatenate([cInit, cEx], axis=0)
#
#Coord2 = np.transpose(np.tile(coord2, reps=[1, len(time2)]))
#Time2 = reshape(np.repeat(time2, repeats=len(coord2)), [len(Coord2), 1])
#input2 = np.concatenate([Time2, Coord2], axis=1)
#cApp = sess.run(model(x), {x:input2})
#cApp = reshape(cApp, [len(time2), len(coord2)])
#
#appErr = []
#plt.figure()
#filepath = '/Users/Riza/Documents/Python/TF_checkpoints/Images'
#filename = []
#for t in range(len(time2)):
#    plt.plot(coord2, cEx[t,:], 'b')
#    plt.xlabel('$x$')
#    plt.ylabel('solution')
#    plt.grid(True)
#    plt.plot(coord2, cApp[t,:], 'r')
#    plt.title('t={0:.2f}s'.format(time2[t]))
#    plt.legend(['exact solution', 'approximate solution'])
#    filename = 'concen' + '{:.2f}'.format(t) + '.png'
#    filepath1 = os.path.join(filepath, filename)
#    plt.savefig(filepath1, dpi=500)
#    plt.show()
#    appErr.append(la.norm(cApp[t,:]-cEx[t,:])/la.norm(cEx[t,:]))
#
#plt.figure()
#plt.plot(time2, appErr)
#plt.xlabel('t')
#plt.ylabel('estimation error')
#plt.grid(True)
#plt.show()
#
#print('average normalized concentration error: {:.3}'.format(np.mean(appErr)))








