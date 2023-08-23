import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm as cmap_cm
#import seaborn as sns

class Lattice(object):
    '''
    Args:
    '''
    def __init__(self, atoms, lattice_vectors):
        self.atoms
        self.lattice_vectors

    def get_lattice(self, n_list=6):
        if not isinstance(nk_list, list):
            nk_list = [nk_list for _ in range(len(self.lattice_vectors))]
        return None


class KSpace(object):
    '''
    Args:
        lattice_vectors : A list of vectors that define the unit cell.
    '''
    def __init__(self, lattice_vectors):
        self._lattice_vectors = np.asarray(lattice_vectors)
        self._reciprocal_vectors = 2.0 * np.pi * np.linalg.inv(lattice_vectors).T
        self._d = len(lattice_vectors)

    def path(self, k_points, n_points=100):
        '''
        Generate a linearly interpolated k-mesh path between k-points in 
        fractional coordinates.

        Args:
            k_points : A list of vectors that specify the k-points in 
                fractional coordinates.

            n_points : (Default 100) Number of points between each k_point.
        '''
        spatial_dim = len(k_points[0])
        segments = len(k_points) - 1
        mesh = np.zeros(shape=(segments*n_points, spatial_dim))
        k = 0
        for i in range(segments):
            ka = k_points[i]
            kb = k_points[i+1]
            diff = np.subtract(kb, ka)
            distance = np.sqrt(sum(diff**2))
            if i == segments-1:
                points = np.linspace(start=0, stop=distance, num=n_points, endpoint=True)
            else:
                points = np.linspace(start=0, stop=distance, num=n_points, endpoint=False)
            for n in points:
                mesh[k,...] = ka + (n/distance) * diff
                k += 1
        self._nks = mesh.shape[0]
        self._frac_list = mesh
        self._k_type = 'path'

    def old_hypercubic(self, nk_list=6, shift=False):
        from itertools import product
        if not isinstance(nk_list, list):
            nk_list = [nk_list for _ in range(len(self.lattice_vectors))]
        
        nks = np.prod(nk_list)
        spatial_dim = len(nk_list)
        coords = [range(nk) for nk in nk_list]
        mesh = np.empty(shape=(nks, spatial_dim))
        
        for idx,coord in enumerate(product(*coords)):
            for i in range(len(coord)):
                mesh[idx,i] = coord[i]/nk_list[i]
                if shift:
                    mesh[idx,i] += 0.5/nk_list[i]
        
        self._nks = np.prod(nk_list)
        self._mesh_list = mesh
        self._k_type = 'full_bz'

    def monkhorst_pack(self, nk_list=6, shift_list=0, basis='fractional', domain=[0, 1], endpoint=True):
        '''
        Generate an equally spaced hypercubic k-mesh

        Args: b
            nk_list : A list for how many points in each spatial dimension.
                E.g., in 3D [nkx, nky, nkz], where the length of the list is
                the number of dimensions.

            shift_list : Will shift the mesh by, e.g., shift in each dimension

            basis : Fractional (default) of the reciprocal lattice vectors or Cartesian

        Returns:
            A numpy meshgrid array
        '''
        if (not isinstance(nk_list, list)) and (not isinstance(nk_list, np.ndarray)):
            nk_list = np.array([nk_list for _ in range(self.d)], dtype=int)
        self._nk_list = np.array(nk_list)
        self._mesh_shape = tuple(self._nk_list)
        self._nks = np.prod(nk_list)
        
        if (not isinstance(shift_list, list)) and (not isinstance(shift_list, np.ndarray)):
            shift_list = np.array([shift_list for _ in range(self.d)], dtype=float)
        self._shift_list = np.array(shift_list)
        
        self._basis = basis
        self._domain = domain
        self._endpoint = endpoint
        self._k_type = 'full_bz'

        ks = [np.linspace(*self._domain, self._nk_list[i], endpoint=self._endpoint) 
              + self._shift_list[i] for i in range(self.d)]
              #+ self._shift_list[i]/self._nk_list[i] for i in range(self.d)]
        if self.basis == 'fractional':
            self._frac_mesh = np.array( np.meshgrid(*ks, indexing='xy') ) # 'ij'
        elif self.basis == 'cartesian':
            self._cart_mesh = np.array( np.meshgrid(*ks, indexing='xy') ) # 'ij'
        
    def to_mesh(self, A, A_type='kvec'): 
        # For k-space vectors
        #if A.shape == (self._nks, self._d):
        if A_type == 'kvec':
            return np.reshape(A.T, (self.d, *self.mesh_shape))
        # For objects (scalars, matrices, arrays) evaluated on the grid
        #elif A.shape[0] == (self._nks):
        elif A_type == 'array':
            return np.reshape(A, (*self.mesh_shape, *A.shape[1:]))
        else:
            raise ValueError(f'shape {A.shape} of A does not match any known shape !')

    def to_list(self, A, A_type='kvec'):
        # For k-space vectors
        #if A.shape == (self._d, *self._mesh_shape):
        if A_type == 'kvec':
            return np.reshape(A, (self.d,-1)).T
        # For objects (scalars, matrices, arrays) evaluated on the grid
        #elif A.shape[:len(self._mesh_shape)] == self._mesh_shape:
        elif A_type == 'array':
            return np.reshape(A, (self.nks, *A.shape[len(self.mesh_shape):]))
        else:
            raise ValueError(f'shape {A.shape} of A does not match any known shape !')
    
    def transform_ks(self, ks, to_basis):
        # G is the recirpocal vectors in row format, so we first transpose it so that b1,b2,b3 are the columns of the matrix
        # But k_frac is also a list of vectors and is not in columns, so we first have to transpose it
        # Then, k_cart = G.T k_frac.T so it is a matrix going into a column array
        # To allow proper broadcasting for numpy we can instead do k_cart = (k_frac G ) because Ax = (x.T A.T).T
        if to_basis == 'cartesian':
            return  ks @ self._reciprocal_vectors
        elif to_basis == 'fractional':
            return ks @ np.linalg.inv(self._reciprocal_vectors) # valid because inv(A.T).T = inv(A)
        else:
            raise ValueError('Unrecognized coordinate system !')
    
    def mesh(self, basis):
        if basis == 'cartesian':
            return self.cart_mesh
        elif basis == 'fractional':
            return self.frac_mesh
        else:
            raise ValueError('Unrecognized coordinate system !')

    @property
    def basis(self):
        return self._basis

    @property
    def d(self):
        return self._d
    
    @property
    def mesh_shape(self):
        return self._mesh_shape
    
    @property
    def nks(self):
        return self._nks
    
    @property
    def lattice_vectors(self):
        return self._lattice_vectors

    @property
    def reciprocal_vectors(self):
        return self._reciprocal_vectors

    @property
    def vc(self):
        # FIXME what about arbitrary d?
        if self.d == 1:
            return np.abs(self.lattice_vectors)
        elif self.d == 2:
            return np.abs(np.cross(*self.lattice_vectors))
        elif self.d == 3:
            return np.dot(self.lattice_vectors[0], np.cross(*self.lattice_vectors[1:]) )
        else:
            raise ValueError('Greater that d=3 dimensions is not implemented !')

    @property
    def vg(self):
        return (2*np.pi)**self._d / self.vc
        
    @property
    def frac_list(self):
        if not hasattr(self,'_frac_list') and (hasattr(self,'_cart_list') or hasattr(self,'_cart_mesh')):
            # k_frac = G^-1 k_cart
            self._frac_list = self._transform_ks(self._cart_list, to_basis='fractional')
        if not hasattr(self,'_frac_list'):
            self._frac_list = self.to_list(self.frac_mesh)
        return self._frac_list
    
    @property
    def frac_mesh(self):
        if not hasattr(self,'_frac_mesh'):
            self._frac_mesh = self.to_mesh(self.frac_list)
        return self._frac_mesh
    
    @property 
    def cart_list(self):
        if not hasattr(self,'_cart_list') and (hasattr(self,'_frac_list') or hasattr(self,'_frac_mesh')):
            self._cart_list = self.transform_ks(self.frac_list, to_basis='cartesian')
        elif not hasattr(self,'_cart_list') and hasattr(self,'_cart_mesh'):
            self._cart_list = self.to_list(self.cart_mesh)
        return self._cart_list

    @property 
    def cart_mesh(self):
        if not hasattr(self,'_cart_mesh'):
            self._cart_mesh = self.to_mesh(self.cart_list)
        return self._cart_mesh