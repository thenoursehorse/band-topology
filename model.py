import numpy as np
from itertools import product

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
        self.lattice_vectors = lattice_vectors
        self.reciprocal_vectors = 2.0 * np.pi * np.linalg.inv(lattice_vectors).T

        self._Hks = None
        self._Eks = None
        self._Vks = None

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
            points = np.linspace(start=0, stop=distance, num=n_points, endpoint=True)
            for n in points:
                k_mesh[k,...] = ka + (n/distance) * diff
                k += 1
        return k_mesh

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
        
        return k_mesh
    
    def get_Hks(self, Hks_fnc, k_mesh):
        # ks = G x, but x is a list of vectors, to vectorize the operation we
        # can do instead ks = (k_mesh x.T) because Ax = (x.T A.T).T
        ks = np.dot(k_mesh, self.reciprocal_vectors.T)
        self._Hks = Hks_fnc(ks)
        self._Eks, self._Vks = np.linalg.eigh(self._Hks)
        return self._Hks

    #FIXME
    def plot(self):
        return None

    def run(self, Hks_fnc, k_mesh=None):
        if k_mesh is None:
            k_mesh = self.hypercubic_k_mesh()
        self.get_Hks(Hks_fnc, k_mesh=k_mesh)

    @property
    def Hks(self):
        return self._Hks

    @property
    def Eks(self):
        return self._Eks

    @property
    def Vks(self):
        return self._Vks

def cubic_kin(ks, t=1, a=1, orb_dim=1):
    di = np.diag_indices(orb_dim)
    nks = ks.shape[0]
    Hks = np.zeros([nks, orb_dim, orb_dim])
    Hks[:,di[0],di[1]] = -2.0 * t * np.sum(np.cos(a * ks), axis=1)[:, None]
    return Hks

cubic_lattice_vectors = [[1,0,0],[0,1,0],[0,0,1]]
cubic = KSpace(lattice_vectors=cubic_lattice_vectors)
cubic.run(Hks_fnc=cubic_kin)

square_lattice_vectors = [[1,0],[0,1]]
square = KSpace(lattice_vectors=square_lattice_vectors)
square_high_symmetry_points = {'G':[0,0], 'X':[1,0], 'M':[1,1]}
k_path = square.path_k_mesh(k_points=list(square_high_symmetry_points.values()), n_points=100)
square.run(Hks_fnc=cubic_kin, k_mesh=k_path)