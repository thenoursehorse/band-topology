import numpy as np
from itertools import product
import matplotlib.pyplot as plt

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


class Plotter(object):
    def __init__(self, Eks):
        self._Eks = Eks


class KSpace(object):
    '''
    Args:
        lattice_vectors : A list of vectors that define the unit cell.
    '''
    def __init__(self, lattice_vectors):
        self.lattice_vectors = np.asarray(lattice_vectors)
        self.reciprocal_vectors = 2.0 * np.pi * np.linalg.inv(lattice_vectors).T

        self._k_mesh = None
        self._nks = None
        self._nbands = None

        self._Hks = None
        self._Eks = None
        self._Vks = None
        
        self._k_type = None

    def path_k_mesh(self, k_points, n_points=100):
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
        k_mesh = np.zeros(shape=(segments*n_points, spatial_dim))
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
                k_mesh[k,...] = ka + (n/distance) * diff
                k += 1
        self._k_mesh = k_mesh
        self._k_type = 'path'
        return self._k_mesh

    def hypercubic_k_mesh(self, nk_list=6, shift=False):
        '''
        Generate an equally spaced hypercubic k-mesh in fractional coordinates.

        Args: b
            nk_list : A list for how many points in each spatial dimension.
                E.g., in 3D [nkx, nky, nkz], where the length of the list is
                the number of dimensions.

            shift : (Default False) If True will shift the mesh by half 
                a space.

        Returns:
            A 2-dim array of the mesh value where the first index is a linear 
            index labelling the k-mesh, and the second index is the dimension.
        '''
        if not isinstance(nk_list, list):
            nk_list = [nk_list for _ in range(len(self.lattice_vectors))]
        
        nks = np.prod(nk_list)
        spatial_dim = len(nk_list)
        coords = [range(nk) for nk in nk_list]
        k_mesh = np.empty(shape=(nks, spatial_dim))
        
        for idx,coord in enumerate(product(*coords)):
            for i in range(len(coord)):
                k_mesh[idx,i] = coord[i]/nk_list[i]
                if shift:
                    k_mesh[idx,i] += 0.5/nk_list[i]
        
        self._k_mesh = k_mesh
        self._k_type = 'full_bz'
        return self._k_mesh
    
    def get_Hks(self, Hks_fnc, k_mesh):
        #ks = np.zeros(shape=k_mesh.shape)
        #for k in range(ks.shape[0]):
        #    ks[k] = np.dot(self.reciprocal_vectors, k_mesh[k])
        # ks = G x, but x is a list of vectors, to vectorize the operation we
        # can do instead ks = (k_mesh x.T) because Ax = (x.T A.T).T
        ks = np.dot(k_mesh, self.reciprocal_vectors.T)
        self._Hks = Hks_fnc(ks, self.lattice_vectors)
        self._Eks, self._Vks = np.linalg.eigh(self._Hks)
        self._nks = self._Eks.shape[0]
        self._nbands = self._Eks.shape[1]
        return self._Hks

    def plot_full_bz(self, ax=None):
        return None

    #FIXME
    def plot_path(self, ax=None):
        x = [i for i in range(self.nks)]
        for n in range(self.nbands):
            plt.plot(x, self.Eks[:,n])
        plt.show()

    def plot(self):
        if self._k_type == 'path':
            self.plot_path()
        else:
            self.plot_full_bz()

    def run(self, Hks_fnc, k_mesh=None):
        if k_mesh is None:
            k_mesh = self.hypercubic_k_mesh()
        self.get_Hks(Hks_fnc, k_mesh=k_mesh)

    @property
    def k_mesh(self):
        return self._k_mesh

    @property
    def nks(self):
        return self._nks
    
    @property
    def nbands(self):
        return self._nbands
    
    @property
    def Hks(self):
        return self._Hks

    @property
    def Eks(self):
        return self._Eks

    @property
    def Vks(self):
        return self._Vks

def cubic_kin(ks, lattice_vectors, t=1, orb_dim=1):
    di = np.diag_indices(orb_dim)
    nks = ks.shape[0]
    Hks = np.zeros([nks, orb_dim, orb_dim])
    Hks[:,di[0],di[1]] = -2.0 * t * np.sum(np.cos(ks), axis=1)[:, None]
    return Hks

cubic_lattice_vectors = [[1,0,0],[0,1,0],[0,0,1]]
cubic = KSpace(lattice_vectors=cubic_lattice_vectors)
cubic.run(Hks_fnc=cubic_kin)

square_lattice_vectors = [[1,0],[0,1]]
square = KSpace(lattice_vectors=square_lattice_vectors)
square_high_symmetry_points = {'G':[0,0], 'X':[1,0], 'M':[1,1], 'G2':[0,0]}
k_path = square.path_k_mesh(k_points=list(square_high_symmetry_points.values()))
square.run(Hks_fnc=cubic_kin, k_mesh=k_path)
square.plot()

def honeycomb_kin(ks, lattice_vectors, t=1):
    a1 = lattice_vectors[0]
    a2 = lattice_vectors[1]
    a3 = - (a1 + a2)
    
    n1 = (a2 - a1) / 3
    n2 = (a3 - a2) / 3
    n3 = (a1 - a3) / 3
    ni = [n1, n2, n3]
    
    gamma = 0
    for i in range(len(ni)):
        gamma += np.exp( 1j * np.dot(ks, ni[i]) )
    
    orb_dim = 2
    nks = ks.shape[0]
    Hks = np.zeros([nks, orb_dim, orb_dim], dtype=np.complex_)
    Hks[:,0,1] = gamma
    Hks[:,1,0] = gamma.conj()
    return Hks

hexagonal_lattice_vectors = [[-0.5, np.sqrt(3)/2], [-0.5, -np.sqrt(3)/2]]
hexagonal_high_symmetry_points = {'G':[0,0], 'M':[1.0/2.0,0], 'K':[1.0/3.0,1.0/3.0], 'G2':[0,0]}
honeycomb = KSpace(lattice_vectors=hexagonal_lattice_vectors)
k_path = honeycomb.path_k_mesh(k_points=list(hexagonal_high_symmetry_points.values()))
honeycomb.run(Hks_fnc=honeycomb_kin, k_mesh=k_path)
honeycomb.plot()