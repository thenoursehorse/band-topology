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

class KSpacePath(object):
    '''
    This is a helper class to hold things needed for the paths
    '''
    def __init__(self, special_points, n_points):
        self._special_points = special_points
        self._n_points = n_points
        self.make_path()
            
    def make_path(self):
        special_points_keys = list( self.special_points.keys() )
        special_points_list = list(self._special_points.values())
        # Make sure are numpy arrays
        for i in range(len(special_points_list)):
            special_points_list[i] = np.array( special_points_list[i] )
        dim = len(special_points_list[0])
        n_segments = len(special_points_list) - 1

        # Find distances between points
        distances = []
        differences = []
        for i in range(n_segments):
            ka = special_points_list[i]
            kb = special_points_list[i+1]
            differences.append( np.subtract(kb, ka) )
            distances.append( np.sqrt(sum(differences[i]**2)) )
        total_distance = sum(distances)

        # Work out how many points in each segment based on length of segment
        self._nks_segment = []
        for i in range(n_segments):
            self._nks_segment.append( int( self._n_points * distances[i]/total_distance) )
        self._nks = sum(self._nks_segment)

        self._ks_list = np.zeros(shape=(self._nks, dim))
        counter = 0
        for i in range(n_segments):
            if i == n_segments-1:
                endpoint = True
            else:
                endpoint = False
            points = np.linspace(start=0, stop=distances[i], num=self._nks_segment[i], endpoint=endpoint)
            # special_points_list[i].shape = (dim), points.shape = (nks_segment[i]), differences.shape = (dim)
            # So add a fake dimension to points so differences gets element wise multiplied into points
            # Then add fake dimension to special_points_list to broadcast addition properly
            out = special_points_list[i][None,...] + points[...,None] * differences[i] / distances[i]
            if i == 0:
                self._ks_list[:self._nks_segment[i]] = out
            else:
                self._ks_list[counter:counter+self._nks_segment[i]] = out
            counter += self._nks_segment[i]

        # Work out which index in self._ks_list corresponds to which special point
        self._special_points_indices = dict()
        self._special_points_indices[special_points_keys[0]] = 0
        counter = 0
        for i in range(len(self._nks_segment)-1):
            key = special_points_keys[i+1]
            counter += self._nks_segment[i]
            self._special_points_indices[key] = counter
        self._special_points_indices[special_points_keys[-1]] = self.nks - 1
    
    @property
    def ks_list(self):
        return self._ks_list
    
    @property
    def special_points(self):
        return self._special_points
    
    @property
    def nks_segment(self):
        return self._nks_segment
    
    @property
    def nks(self):
        return self._nks

    @property
    def special_points_indices(self):
        return self._special_points_indices
        
class KSpace(object):
    '''
    Args:
        lattice_vectors : A list of vectors that define the unit cell.
    '''
    def __init__(self, lattice_vectors):
        self._lattice_vectors = np.asarray(lattice_vectors)
        self._reciprocal_vectors = 2.0 * np.pi * np.linalg.inv(lattice_vectors).T
        self._d = len(lattice_vectors)

    def path(self, special_points, n_points=100, basis='fractional'):
        '''
        Generate a linearly interpolated k-mesh path between k-points.

        Args:
            kpts : A dictionary of vectors that specify the k-points in 
                fractional or Cartesian coordinates.

            n_points : (Default 100) Number of total points along entire path

            basis : The coordinate system special_points are given in
        '''

        self._path_class = KSpacePath(special_points=special_points, n_points=n_points)
        self._nks = self._path_class.nks

        if basis == 'cartesian':
            self._cart_list = self._path_class.ks_list
        elif basis == 'fractional':
            self._frac_list = self._path_class.ks_list
        self._k_type = 'path'
        
        # FIXME I think this is fine because it is a 1-D mesh, a line in k-space
        self._mesh_shape = tuple([self.nks])

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
        self._nk_list = nk_list
        self._mesh_shape = tuple(nk_list)
        self._nks = np.prod(nk_list)
        
        if (not isinstance(shift_list, list)) and (not isinstance(shift_list, np.ndarray)):
            shift_list = np.array([shift_list for _ in range(self.d)], dtype=float)
        self._shift_list = np.array(shift_list)
        
        self._basis = basis
        self._domain = domain
        self._endpoint = endpoint
        self._k_type = 'monkhorst'

        ks = [np.linspace(*self._domain, nk_list[i], endpoint=self._endpoint) 
              + self._shift_list[i] for i in range(self.d)]
              #+ self._shift_list[i]/nk_list[i] for i in range(self.d)]
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
            self._frac_list = self.transform_ks(self._cart_list, to_basis='fractional')
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