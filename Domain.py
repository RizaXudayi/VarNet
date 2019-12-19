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
This file provides the classes for definition and sicretization of domains.

"""

#%% Modules:

import math

import numpy as np
shape = np.shape
reshape = np.reshape
size = np.size
import numpy.linalg as la

import matplotlib.pyplot as plt
from matplotlib.path import Path

from UtilityFunc import UF
uf = UF()

#%% Domain class:

class Domain():
    """Domain class defines the domain of the PDE and its boundaries."""
    
    def __init__(self, dim, lim):
        """
        Initialize the attributes of the Domain class. The class is written as
        general as possible and some of the methods need to be modified in 
        appropriate subclasses.
        
        Inputs:
            dim: dimension of the domain
            lim [2 x dim]: limits of each dimension at one column
        """
        self.dim = dim
        self.lim = np.array(lim)
        
        
    def scaleCoord(self, x):
        """
        Function to pre-process the coordinates so that inputs to the NN are all
        between [-1,1]. This requires a centering followed by a scaling determined
        from the limits of each coordinate.
        
        Input:
            x [n x dim]: coordinates of points to be processed
            
        Output:
            z [n x dim]: centered and scaled coordinates
        """
        if shape(x)[1]!=self.dim:
            raise ValueError('Input dimensions are incompatible with domain dimension!')
            
        cen = np.mean(self.lim, axis=0)         # center at each dimension
        scale = np.diff(self.lim, axis=0)       # length at each dimension
        
        return (x-cen)/scale*2
    
    
    def isInside(self, x):
        """
        Function to determine if the coordinates provided are inside the domain.
        Note that this function is domain specific and needs to be redefined in
        a subclass.
        
        Input:
            x [n x dim]: coordinates of points to be checked
            
        Output:
            flag [1 x dim]: logical values indicating whether each point is inside
            the domain
        """
        raise Exception('This function must be redefined in the subclass!')
        
        
    def getMesh(self):
        """
        Retruns a Mesh object corresponding to the mesh generated for the domain.
        Note that this function needs to be redefined in a subclass.
        """
        raise Exception('This function must be redefined in the subclass!')
        
        
#%%
        
class Mesh():
    """Class to store and retrieve attributes of a discretized domain."""
    
    def __init__(self,
                 dim,
                 dof,
                 coordinates,
                 he,
                 bIndNum,
                 bdof,
                 bCoordinates,
                 discNum = [],
                 bDiscNum = []
                 ):
        """
        Initialize the Mesh class to store the attributes of a discretized 
        domain. This class is used to store grid data which are used for 
        training the NN and also ploting the fields. The generation of the 
        actual mesh is done via CAD softwares or other means. A Mesh object
        has the following attributes:
            dim: dimension of domain
            dof: number of discrete points (degrees of freedom)
            coordinates [dof x dim]: coordinates of the discrete points 
                (depending on application might include or exclude the boundary nodes)
            he: element size(s)
            bIndNum: number of boundary indicators in the domain            
            bdof [bIndNum x 1]: number of discrete points on each boundary segment
            bCoordinates [bIndNum x n x dim]: list containing the coordinates of
                discrete points on the boundaries
            discNum [1 x dim]: number of discretization points at each dimension (2D)
            bDiscNum: density of discretization points on the boundaries (2D)
        """
        self.dim = dim
        self.dof = dof
        self.coordinates = coordinates
        self.he = he
        self.bIndNum = bIndNum
        self.bdof = bdof
        self.bCoordinates = bCoordinates
        self.discNum = discNum
        self.bDiscNum = bDiscNum
 
    
#%% Mesh domain class (inherits from Domain class):
    
class MeshDomain(Domain):
    """This class handles complicated domains modeled by a CAD software."""
    
    def __init__(self, meshfile):
        """
        Initializes the class by assigning relevant information to attributes 
        in this class and its superclass.
        
        Input:
            meshfile: filepath to a mesh file containing a dictionary with 
            discretized domain data, among which the followings must exist:
                dim: dimension of domain
                dof: number of discrete points (degrees of freedom)
                coordinates [dof x dim]: coordinates of the discrete points INSIDE domain
                bIndNum: number of boundary indicators in the domain
                bCoordinates [bIndNum x n x dim]: list containing the coordinates of
                    discrete points on the boundaries
        """
        mesh = np.load(meshfile)
        coord = mesh.coordinates
        lim = np.array([[np.min(coord, axis=0)], [np.max(coord, axis=0)]])
        self.coordinates = coord
        self.bIndNum = mesh.bIndNum
        self.boundaryInd = mesh.boundaryInd
        super().__init__(mesh.dim, lim)         # initialize the attributes of the superclass
    
    
    def isInside(self, x):
        """
        Function to determine if the coordinates provided are inside the 
        discretized domain.
        
        Input:
            x [n x dim]: coordinates of points to be checked
            
        Output:
            flag [1 x dim]: logical values indicating whether each point is inside
            the domain
        """
        raise Exception('This function needs to be defined before use!')
        
        
    def getMesh(self):
        """
        Retruns a Mesh object corresponding to the mesh file provided. See Mesh
        class for details.
        """
        raise Exception('This function needs to be defined before use!')


#%% Polygon domain class (inherits from Domain class):
    
class PolygonDomain2D(Domain):
    """
    Class definition for simple 2D domain defined by a polygon.
    Particularly, the vertices of the domain and obstacles inside it are given 
    in matrices, where each row contains one vertex.
    """
    
    def __init__(self,
                 vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
                 obsVertices = [],
                 ):
        """
        Initializes the attributes of the class. The default domain is a 1x1 square.
        
        Inputs:
            vertices [vertexNum x dim]: vertices of the domain (one vertex per row)
            obsVertices [obsNum x obsVerNum x dim]: list of matrices containing
                the vertices of obstacles (default: [])
        
        Attributes:
            dim: domain dimension
            lim [2 x dim]: domain limits
            vertexNum: number of vertices of the domain frame
            vertices
            obsNum: number of obstacles
            obsVertices
            bIndNum: number of boundary segments
            boundryGeom [bIndNum x2x2]: limits for boundary edges
            measure: area of the polygonal domain
        """
        dim = 2                                 # domain dimension (fixed)
        if shape(vertices)[1]!=dim:
            raise ValueError('Vertex dimensions are incompatible with domain dimension!')
        elif type(obsVertices) is not list:
            raise ValueError('obstacle polygons must be given as a list of matrices!')
        
        # Domain limits:
        lim = np.vstack([np.min(vertices, axis=0), np.max(vertices, axis=0)])
        
        # Boundary indicator count:
        obsNum = len(obsVertices)               # number of obstacles
        bIndNum = shape(vertices)[0]            # number of boundary pieces on the domain frame
        bGeom = self.boundaryLims(vertices)
        for i in range(obsNum):                 # loop over obstacles
            bIndNum += shape(obsVertices[i])[0] # number of boundary pieces on obstacle frames
            bGeom = uf.vstack([ bGeom, self.boundaryLims(obsVertices[i]) ])
        
        # Initialize the attributes:
        super().__init__(dim, lim)
        self.vertexNum = shape(vertices)[0]     # number of vertices of the domain
        self.vertices = vertices
        self.obsNum = obsNum                    # number of obstacles in the domain
        self.obsVertices = obsVertices
        self.bIndNum = bIndNum
        self.boundryGeom = bGeom                # limits for boundary edges
        self.measure = uf.polyArea(vertices)    # area of the polygonal domain
        
    

    def boundaryLims(self, vertices):
        """
        Function to construct the limits for the boundaries of a polygon 
        specified by 'vertices'. It works by grouping the corners of the polygon
        into 2x2 matrices where each row contains coordinates of corner.
        Output [bIndNum x 2 x 2]: array containing the coordinates of the corners
            of each edge in one row
        """
        bIndNum = len(vertices)
        vertices = np.repeat(vertices, repeats=2, axis=0)
        vertices = np.vstack([vertices[1:,:], vertices[0,:]])
        return reshape(vertices, newshape=[bIndNum,2,2])
        
        

    def domPlot(self, addDescription=True, figNum=None, frameColor=None):
        """
        Function to plot the domain boundaries.
        
        Input:
            addDescription: add figure labels and details
            figNum: figure number to draw on
            frameColor: color of the outer domain
        """
        
        if uf.isnone(frameColor):
            frameColor = ['r', 'b']
        else:
            frameColor = [frameColor]*2
        
        if uf.isnone(figNum):
            plt.figure()
        else:
            plt.figure(figNum)
        
        # Plot the outer boundaries of the domain:
        vertices = self.vertices
        vertices = np.vstack([vertices, vertices[0,:]])         # close the frame loop
        plt.plot(vertices[:,0], vertices[:,1], frameColor[0])
        plt.axis('scaled')
        Legend = []
        Legend.append('outer boundaries')
        
        # Plot the obstacles:
        for i in range(self.obsNum):
            vertices = self.obsVertices[i]
            vertices = np.vstack([vertices, vertices[0,:]])     # close the frame loop
            plt.plot(vertices[:,0], vertices[:,1], frameColor[1])
            Legend.append('obstacle {}'.format(i+1))
        
        if addDescription:
            plt.title('domain')
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.legend(Legend)
            plt.show()
    
    
    def meshPlot(self, mesh):
        """Function to plot the generated mesh."""
        
        # Plot the outer boundaries of the domain:
        plt.plot(mesh.coordinates[:,0], mesh.coordinates[:,1], 'r.', markersize=2)
        Legend = []
        Legend.append('domain interior')
        
        # Plot the obstacles:
        for i in range(self.bIndNum):
            coord = mesh.bCoordinates[i]
            plt.plot(coord[:,0], coord[:,1], '.', markersize=4)
            Legend.append('boundary {}'.format(i+1))
        
        plt.title('discretized domain')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(Legend)
        plt.show()
        
    
    def isInside(self, x, tol=0.):
        """
        Function to determine if the coordinates provided are inside the 
        discretized domain. The function utilizes 'matplotlib.path.contains_points'
        to perform the desired operation.
        
        Input:
            x [n x dim]: coordinates of points to be checked
            tol: tolerance to make sure that the points do not lie on the boundaries
            
        Output:
            flag [n x 1]: logical values indicating whether each point is inside
            the domain
        """
        if shape(x)[1]!=self.dim:
            raise ValueError('Vertex dimensions are incompatible with domain dimension!')
        
        # Check whether the points are in the domain:
        pth = Path(self.vertices, closed=False)
        inDom = pth.contains_points(x, radius=tol)
        
        if self.obsNum==0:
            return inDom
            
        # Check if the points are on obstacles:
        inObs = np.zeros([shape(x)[0], self.obsNum], dtype=bool)
        for obs in range(self.obsNum):
            pth = Path(self.obsVertices[obs], closed=False)
            inObs[:,obs] = pth.contains_points(x, radius=-tol)
        
        return inDom*np.logical_not(np.prod(inObs, axis=1))
        
    
    def getMesh(self, discNum=100, bDiscNum=50, rfrac=0, sortflg=True, discTol=None):
        """
        This function creates a structured mesh of the polygon domain and discards
        points that fall within obstacles or outside the domain boundaries.
        
        Inputs:
            discNum [1 x dim]: number of elements at each dimension
            bDiscNum [obsNum x 1]: density of elements per unit length for each
                boundary segment
            rfrac: fraction of points to be drawn randomly from a uniform distribution
            sortflg: if True sort the randomly drawn samples
            discTol: minimum distance of discretization points from the boundaries
        
        It retruns a Mesh object corresponding to the mesh file provided.  See Mesh
        class for details.
        """
        
        # Discretize domain interior:
        he, coordinates = self.innerDisc(discNum, rfrac, sortflg, discTol)
        dof = shape(coordinates)[0]
        
        # Discretize domain boundaries:
        bIndNum = shape(self.vertices)[0]                       # number of boundary pieces on the domain frame
        bdof, bCoordinates = self.boundaryDisc(self.vertices,   # outer domain boundaries
                                               bDiscNum, rfrac, sortflg)
        
        for i in range(self.obsNum):                            # loop over obstacles
            bIndNum += shape(self.obsVertices[i])[0]
            bdofTmp, bCoordTmp = self.boundaryDisc(self.obsVertices[i], bDiscNum, rfrac, sortflg)
            bdof.extend(bdofTmp)                                # attach dof of current boundary segments
            bCoordinates.extend(bCoordTmp)                      # attach the coordinates of current boundary segments
        
        
        mesh = Mesh(dim = 2,
                    dof = dof,
                    coordinates = coordinates,                  # coordinates of inner domain
                    he = he,
                    bIndNum = bIndNum,                          # total number of boundary indicators
                    bdof = bdof,
                    bCoordinates = bCoordinates,                # boundary node coordinates
                    discNum = discNum,
                    bDiscNum = bDiscNum)
        return mesh
    
    
    def innerDisc(self, discNum, rfrac=0., sortflg=True, discTol=None):
        """
        Function to discretize the inside of the domain.
        
        Input:
            discNum [1 x dim]: number of elements at each dimension
            rfrac: fraction of points to be drawn randomly from a uniform distribution
            sortflg: if True sort the randomly drawn samples
            discTol: minimum distance of discretization points from the boundaries
        """
        # Error handling:
        if not (np.size(discNum)==1 or np.size(discNum)==2):
            raise ValueError('\'discNum\' dimension incompatible!')
        elif np.size(discNum)==1:
            discNum = [discNum, discNum]
            
        if not uf.isnone(discTol) and not (np.size(discTol)==1 or np.size(discTol)==2):
            raise ValueError('\'discTol\' dimension incompatible!')
        elif not uf.isnone(discTol) and np.size(discTol)==1:
            discTol = [discTol, discTol]
            
        lim = self.lim
        rfrac = rfrac**0.5                                                  # fraction of random samples per dimension
        coord = []
        he = []
        for d in range(2):
            dof = discNum[d]                                                # dof for current dimension
            h = (lim[1,d]-lim[0,d])/(dof+1)                                 # element size
            he.append(h)
            if uf.isnone(discTol):  tol = h
            else:                   tol = np.asscalar(discTol[d])
            
            dof1 = math.floor(dof*rfrac)                                    # random grid
            dof2 = dof - dof1                                               # uniform grid
            coord1 = np.random.uniform(lim[0,d]+tol, lim[1,d]-tol, dof1)    # random test function locations
            coord2 = np.linspace(lim[0,d]+tol, lim[1,d]-tol, dof2)          # uniform test function locations
            coordTmp = np.hstack([coord1, coord2])                          # stack together the random and unform discretizations
            if rfrac>0 and sortflg:
                coordTmp = np.sort(coordTmp)                                # sort coordinates in increasing order
            coord.append(coordTmp)                                          # set of discrete points
        he = np.array(he)                                                   # convert to numpy array
        
        ne = np.prod(discNum)
        x_coord = reshape(np.tile(coord[0], discNum[1]), [ne, 1])           # repeat for y-coordinate
        y_coord = reshape(np.repeat(coord[1], discNum[0]), [ne, 1])         # repeat for x-coordinate
        coord = np.hstack([x_coord, y_coord])                               # concatinate the coordinates
        
        # isInDom = self.isInside(coord, tol=min(he))                        # check that the points are in domain
        isInDom = self.isInside(coord)                                      # check that the points are in domain
        coord = coord[isInDom,:]                                            # retain points inside the domain
        return he, coord
    
    
    def boundaryDisc(self, vertices, bDiscNum, rfrac=0, sortflg=True):
        """
        Function to discretize the boundary segments of a polygon.
        
        Inputs:
            vertices: vertices of the polygon
            bDiscNum [1 x dim]: density of elements in unit length of boundary
            rfrac: fraction of points to be drawn randomly from a uniform distribution
            sortflg: if True sort the randomly drawn samples
            
        Output:
            coord: a list containing the coordinates of each boundary in the 
            current polygon (the order is the same as the entered polygon)
        """
        sideNum = shape(vertices)[0]
        vertices = np.vstack([vertices, vertices[0,:]])     # close the frame loop
        verDiff = np.diff(vertices, axis=0)                 # vertex edge delta
        sideLen = la.norm(verDiff, axis=1)                  # side lengths
        bdof = []
        coord = []
        for i in range(sideNum):                            # loop over sides of the polygon
            discNum = math.ceil(bDiscNum*sideLen[i])        # number of discretization points
            bdof.append(discNum)
            
            dof1 = math.floor(discNum*rfrac)                # random grid
            dof2 = discNum - dof1                           # uniform grid
            step1 = np.random.uniform(size=dof1)            # random steps
            step2 = np.linspace(0.0, 1.0, num=dof2)         # uniform steps
            step = np.hstack([step1, step2])                # stack together the random and unform discretizations
            if rfrac>0 and sortflg:
                step = np.sort(step)                        # sort coordinates in increasing order
            step = np.tile(step, [2, 1]).T                  # steps along the boundary
            coord.append(vertices[i,:] + verDiff[i,:]*step) # coordinates
        
        return bdof, coord
    
    
    
    def meshTot(self, discNum=100, rfrac=0., sortflg=True):
        """
        This function creates a structured mesh of the polygon domain and discards
        points that fall within obstacles or outside the domain boundaries.
        The difference with the getMesh() function is that this function generates 
        a mesh for whole domain including the boundaries whereas getMesh() 
        considers the boundaries and inner domain separately.
        
        Inputs:
            discNum [1 x dim]: number of elements at each dimension
            rfrac: fraction of points to be drawn randomly from a uniform distribution
            sortflg: if True sort the randomly drawn samples
        
        It retruns a Mesh object corresponding to the mesh file provided.  See Mesh
        class for details.
        """
        # Error handling:
        if not (np.size(discNum)==1 or np.size(discNum)==2):
            raise ValueError('\'discNum\' dimension incompatible!')
        elif np.size(discNum)==1:
            discNum = [discNum, discNum]
            
        lim = self.lim
        rfrac = rfrac**0.5                                              # fraction of random samples per dimension
        coord = []
        he = []
        for d in range(2):
            dof = discNum[d]                                            # dof for current dimension
            h = (lim[1,d]-lim[0,d])/(dof+1)                             # element size
            he.append(h)
            
            dof1 = math.floor(dof*rfrac)                                # random grid
            dof2 = dof - dof1                                           # uniform grid
            coord1 = np.random.uniform(lim[0,d], lim[1,d], dof1)        # random test function locations
            coord2 = np.linspace(lim[0,d], lim[1,d], dof2)              # uniform test function locations
            coordTmp = np.hstack([coord1, coord2])                      # stack together the random and unform discretizations
            if rfrac>0 and sortflg:
                coordTmp = np.sort(coordTmp)                            # sort coordinates in increasing order
            coord.append(coordTmp)                                      # set of discrete points
        he = np.array(he)                                               # convert to numpy array
        
        ne = np.prod(discNum)
        x_coord = reshape(np.tile(coord[0], discNum[1]), [ne, 1])       # repeat for y-coordinate
        y_coord = reshape(np.repeat(coord[1], discNum[0]), [ne, 1])     # repeat for x-coordinate
        coord = np.hstack([x_coord, y_coord])                           # concatinate the coordinates
        
        isInDom = self.isInside(coord, tol=1e-3)                        # check that the points are in domain
        coord = coord[isInDom,:]                                        # retain points inside the domain
        dof = len(coord)
        
        mesh = Mesh(dim = 2,
                    dof = dof,
                    coordinates = coord,                                # coordinates of inner domain
                    he = he,
                    bIndNum = 0,                                        # total number of boundary indicators
                    bdof = None,
                    bCoordinates = None,                                # boundary node coordinates
                    discNum = discNum,
                    bDiscNum = None)
        return mesh

#%% 1D domain class (inherits from Domain class):
    
class Domain1D(Domain):
    """
    Class definition for 1D domain defined by an interval.
    """
    
    def __init__(self, interval = np.array([-1.0, 1.0]) ):
        """
        Initializes the attributes of the class. The default domain is [-1,1] interval.
        
        Inputs:
            interval [1x2]: domain interval
        
        Attributes:
            dim: domain dimension
            lim [2 x 1]: domain limits
            bIndNum: number of boundary segments
            measure: domain length
        """
        dim = 1                 # domain dimension (fixed)
        if np.size(shape(interval))!=dim:
            raise ValueError('interval must be a vector!')
        
        # Domain limits:
        lim = reshape(interval, [2,1])
        
        # Initialize the attributes:
        super().__init__(dim, lim)
        self.bIndNum = 2                        # boundary indicator count
        self.measure = interval[1]-interval[0]  # domain length
    
    
    def isInside(self, x, tol=0.):
        """
        Function to determine if the coordinates provided are inside the 
        discretized domain.
        
        Input:
            x [nx1]: coordinates of points to be checked
            tol: tolerance to make sure that the points do not lie on the boundaries
            
        Output:
            flag [nx1]: logical values indicating whether each point is inside
            the domain
        """
        if shape(x)[1]!=self.dim:
            raise ValueError('Vertex dimensions are incompatible with domain dimension!')
        
        # Check whether the points are in the domain:
        lim = self.lim
        
        return (lim[0]+tol<=x)*(x<=lim[1]-tol)
        
    
    def getMesh(self, discNum=100, bDiscNum=None, rfrac=0, sortflg=True, discTol=None):
        """
        This function creates a structured mesh of the polygon domain and discards
        points that fall within obstacles or outside the domain boundaries.
        
        Inputs:
            discNum: number of elements
            bDiscNum: dummy variable added for universality of the code
            rfrac: fraction of points to be drawn randomly from a uniform distribution
            sortflg: if True sort the randomly drawn samples
            discTol: minimum distance of discretization points from the boundaries
        
        It retruns a Mesh object corresponding to the mesh file provided.  See Mesh
        class for details.
        """
        # Error handling:
        if size(discNum)==0 or size(discNum)>1:
            raise ValueError('number of discretization points must be a scalar!')
        elif size(discNum)==1 and shape(discNum)!=():
            discNum = discNum[0]
        
        # Project the random sampling fraction back to [0,1] interval:
        if rfrac<0:      rfrac = 0
        elif rfrac>1:    rfrac = 1
        
        # Data:
        dim = self.dim
        dof = discNum
        lim = self.lim
        
        # Discretize domain INTERIOR:
        he = (lim[1]-lim[0])/(discNum+1)                                # element size
        if uf.isnone(discTol):  tol = he                                # discretization tolerance
        else:                   tol = np.asscalar(discTol)
        dof1 = math.floor(dof*rfrac)                                    # random grid
        dof2 = dof - dof1                                               # uniform grid
        
        coord1 = np.random.uniform(lim[0]+tol, lim[1]-tol, dof1)        # random test function locations
        coord2 = np.linspace(lim[0]+tol, lim[1]-tol, dof2)              # uniform test function locations
        coordinates = uf.hstack([coord1, coord2])                       # stack together the random and unform discretizations
        if rfrac>0 and sortflg:
            coordinates = np.sort(coordinates)                          # sort coordinates in increasing order
        coordinates = np.reshape(coordinates, [dof, 1])                 # store in column vector
        
        # Discretize domain boundaries:
        bIndNum = 2                                                     # number of boundary pieces on the domain frame
        bdof = np.ones(bIndNum, dtype=int)
        bCoordinates = reshape(lim, [bIndNum, 1, dim])
        
        mesh = Mesh(dim = dim,
                    dof = dof,
                    coordinates = coordinates,                          # coordinates of inner domain
                    he = he,
                    bIndNum = bIndNum,                                  # total number of boundary indicators
                    bdof = bdof,
                    bCoordinates = bCoordinates,                        # boundary node coordinates
                    discNum = discNum)
        return mesh
    
    
    def meshTot(self, discNum=100):
        """
        This function creates a structured mesh of the polygon domain and discards
        points that fall within obstacles or outside the domain boundaries.
        The difference with the getMesh() function is that this function generates 
        a mesh for whole domain including the boundaries whereas getMesh() 
        considers the boundaries and inner domain separately.
        
        Inputs:
            discNum: number of elements
        
        It retruns a Mesh object corresponding to the mesh file provided.  See Mesh
        class for details.
        """
        # Error handling:
        if size(discNum)==0 or size(discNum)>1:
            raise ValueError('number of discretization points must be a scalar!')
        elif size(discNum)==1 and shape(discNum)!=():
            discNum = discNum[0]
        
        # Data:
        dim = self.dim
        dof = discNum
        lim = self.lim
        
        # Discretize domain interior:
        he = (lim[1]-lim[0])/(discNum-1)                    # element size
        coordinates = np.linspace(lim[0], lim[1], dof)      # test function locations
        coordinates = np.reshape(coordinates, [dof, 1])
        
        # Discretize domain boundaries:
        bIndNum = None
        bdof = None
        bCoordinates = None
        
        mesh = Mesh(dim = dim,
                    dof = dof,
                    coordinates = coordinates,              # coordinates of inner domain
                    he = he,
                    bIndNum = bIndNum,                      # total number of boundary indicators
                    bdof = bdof,
                    bCoordinates = bCoordinates,            # boundary node coordinates
                    discNum = discNum)
        return mesh
    
    






