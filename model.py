import numpy as np

from scipy import integrate

import matplotlib.pyplot as plt
from matplotlib import cm as cmap_cm

from copy import deepcopy

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

    def monkhorst_pack(self, nk_list=6, shift_list=0, basis='fractional', domain=[-0.5, 0.5], endpoint=True):
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
        if basis == 'fractional':
            self._frac_mesh = np.array( np.meshgrid(*ks, indexing='xy') ) # 'ij'
        elif basis == 'cartesian':
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
    def frac_list(self):
        if not hasattr(self,'_frac_list') and (hasattr(self,'_cart_list') or hasattr(self,'_cart_mesh')):
            # k_frac = G^-1 k_frac
            self._frac_list = self._cart_list @ np.linalg.inv(self._reciprocal_vectors).T
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
            # k_cart = G k_frac, but k_frac is a list of vectors not columns.
            # To vectorize the operation we can instead do k_cart = (mesh k_frac.T) because Ax = (x.T A.T).T
            self._cart_list = self.frac_list @ self._reciprocal_vectors.T
        elif not hasattr(self,'_cart_list') and hasattr(self,'_cart_mesh'):
            self._cart_list = self.to_list(self.cart_mesh)
        return self._cart_list

    @property 
    def cart_mesh(self):
        if not hasattr(self,'_cart_mesh'):
            self._cart_mesh = self.to_mesh(self.cart_list)
        return self._cart_mesh
    

class TightBinding():
    def __init__(self, Hks_fnc, kspace_class):
        self._Hks_fnc = Hks_fnc
        self._kspace = kspace_class
        self.set_Hks()

    def set_Hks(self):
        self._Hks = self._Hks_fnc(self._kspace)
        # FIXME check if it is a mesh and change to a list
        #self._Eks, self._Vks = np.linalg.eigh(self._kspace.to_list(self._Hks, A_type='array'))
        #self._Eks = self._kspace.to_mesh(self._Eks, A_type='array')
        #self._Vks = self._kspace.to_mesh(self._Vks, A_type='array')
        # Don't need to do any of above because eigh will always take the 
        # last 2 dimensions for computing the spectrum (and do this for all other indices)
        self._Eks, self._Vks = np.linalg.eigh(self._Hks)
        self._nbands = self._Eks.shape[-1]
        #return self._Hks

    def set_Pks(self):
        if not hasattr(self,'_Vks'):
            self.get_Hks()

        #v1 = self.Vks[0,0][:,0][:,np.newaxis] # make into a column vector, but not needed for outer
        # Several ways to do the same thing
        #P1 = v1 @ v1.conj().T
        #P1 = np.outer(v1, v1.conj())
        #P1 = np.einsum('i,j->ij', v1.ravel(), v1.conj().ravel())
        #P1 = np.multiply.outer(v1.ravel(), v1.conj().ravel())
        #P1 = np.tensordot(v1.ravel(), v1.conj().ravel(), axes=((), ()))
        #check = P1 @ v1 # should give back v1 if v1 is a column vector
        #err = np.linalg.norm(v1 - check)

        # maybe use https://pypi.org/project/opt-einsum/?
        self._Pks = np.einsum('...in,...jn->...nij', self.Vks, self.Vks.conj())
        # What is the equivalent tensordot operations?
        # Do I need np.moveaxis?
        # https://www.reddit.com/r/Python/comments/qevahz/tensor_contractions_with_numpys_einsum_function/
        #self._Pks = np.tensordot(self.Vks, self.Vks.conj(), axis=((-2),(-2)))
    
    #FIXME
    def plot_path(self, ax=None):
        x = [i for i in range(self._kspace.nks)]
        for n in range(self.nbands):
            plt.plot(x, self._Eks[:,n])
        plt.show()

    def plot_contour(self, band=0, cmap=cmap_cm.coolwarm, xlim=None, ylim=None):
        fig, ax = plt.subplots()
        contour = ax.contourf(*self._kspace.cart_mesh/np.pi, self.Eks[...,band], cmap=cmap)
        ax.set_xlabel('$k_x/\pi$')
        ax.set_ylabel('$k_y/\pi$')
        if xlim is not None:
            ax.set_xlim(np.array(xlim)/np.pi)
        if ylim is not None:
            ax.set_ylim(np.array(ylim)/np.pi)
        fig.colorbar(contour)
        plt.show()
    
    def plot_surface(self, band=None, cmap=cmap_cm.coolwarm, xlim=None, ylim=None):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        if band is None:
            vmin = np.min(self.Eks)
            vmax = np.max(self.Eks)
            for n in range(self.nbands):
                surf = ax.plot_surface(*self._kspace.cart_mesh/np.pi, self.Eks[...,n], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            surf = ax.plot_surface(*self._kspace.cart_mesh/np.pi, self.Eks[...,band], cmap=cmap)
        ax.set_xlabel('$k_x/\pi$')
        ax.set_ylabel('$k_y/\pi$')
        if xlim is not None:
            ax.set_xlim(np.array(xlim)/np.pi)
        if ylim is not None:
            ax.set_ylim(np.array(ylim)/np.pi)
        fig.colorbar(surf, shrink=0.5)
        plt.show()

    @property
    def nbands(self):
        return self._nbands
    
    @property
    def Hks(self):
        if not hasattr(self,'_Hks'):
            self.set_Hks()
        return self._Hks

    @property
    def Eks(self):
        if not hasattr(self,'_Eks'):
            self.set_Hks()
        return self._Eks

    @property
    def Vks(self):
        if not hasattr(self,'_Vks'):
            self.set_Hks()
        return self._Vks
    
    @property
    def Pks(self):
        if not hasattr(self,'_Pks'):
            self.set_Pks()
        return self._Pks
    
class Topology:
    def __init__(self, tb):
        self._tb = tb
        self._kspace = tb._kspace

    def kspace_shifted(self, i, delta):
        shift_list = deepcopy(self._kspace._shift_list)
        shift_list[i] = shift_list[i] + delta
        kspace_shifted = KSpace(lattice_vectors=self._kspace.lattice_vectors)
        if self._kspace._k_type == 'full_bz':
            # FIXME it has to be shifted in Cartesian coordinates, otherwise
            # the conversion from frac to cart will make the shift not entirely along
            # kx or ky.
            # So I have to work out shift and domain in appropriate coordinates
            # and then convert
            kspace_shifted.monkhorst_pack(nk_list=self._kspace._nk_list,
                                          shift_list=shift_list,
                                          basis=self._kspace._basis, 
                                          #basis='cartesian',
                                          domain=self._kspace._domain, 
                                          endpoint=self._kspace._endpoint)
        else:
            raise ValueError('Unrecognized kspace grid type !')
        return kspace_shifted
    
    @staticmethod
    def check_projector(Pks, dPks, i, tol=1e-8):
        err = np.linalg.norm(dPks @ Pks - Pks @ dPks)
        if err > tol:
            print(f'WARNING: The projectors and their derivatives on dim {i} are bad with error {err} !')
    
    def get_dPks(self, i=0, delta=1e-4):
        ks_up = self.kspace_shifted(i, delta)
        ks_dn = self.kspace_shifted(i, -delta)
        tb_up = TightBinding(Hks_fnc=self._tb._Hks_fnc, kspace_class=ks_up)
        tb_dn = TightBinding(Hks_fnc=self._tb._Hks_fnc, kspace_class=ks_dn)
        return (tb_up.Pks - tb_dn.Pks) / (2.0*delta)
        
        #if i == 0:
        #    ks = self._kspace.cart_mesh[0][0]
        #if i == 1:
        #    ks = self._kspace.cart_mesh[1][:,0]
        #return np.gradient(self._tb.Pks, ks, axis=i)
    
    def set_dPks(self, delta=1e-4):
        self._dPks = [self.get_dPks(i=i, delta=delta) for i in range(self._kspace.d)]
    
    def geometric_tensor(self, subspace=[0], i=0, j=0, delta=1e-4):
        if not hasattr(self,'_dPks'):
            self.set_dPks(delta=delta)
        
        Pks = 0
        dPks_i = 0
        dPks_j = 0
        for n in subspace:
            # The band index is always in this position
            Pks = self._tb.Pks[...,n,:,:]
            dPks_i += self._dPks[i][...,n,:,:]
            dPks_j += self._dPks[j][...,n,:,:]
        
        # FIXME the derivatives always suck, I need to make them better
        #self.check_projector(Pks, dPks_i, i)
        #self.check_projector(Pks, dPks_j, j)

        #return 2.0 * np.trace(dPks_i @ dPks_j - dPks_i @ Pks @ dPks_j, axis1=-2, axis2=-1)
        return 2.0 * np.trace(Pks @ dPks_i @ dPks_j, axis1=-2, axis2=-1)
    
    def quantum_metric(self, subspace=[0], i=0, j=0, delta=1e-4, recalculate=False):
        if not hasattr(self,'_dPks'):
            self.set_dPks(delta=delta)

        if recalculate:
            Pks = 0
            dPks_i = 0
            dPks_j = 0
            for n in subspace:
                # The band index is always in this position
                Pks = self._tb.Pks[...,n,:,:]
                dPks_i += self._dPks[i][...,n,:,:]
                dPks_j += self._dPks[j][...,n,:,:]

            self.check_projector(Pks, dPks_i, i)
            self.check_projector(Pks, dPks_j, j)

            g =  0.5 * np.trace(dPks_i @ dPks_j, axis1=-2, axis2=-1)
            if np.linalg.norm(g.imag) > 1e-10:
                print('WARNING: The quantum metric has non-neglibile imaginary parts !')
            return g.real
        else:
            return self.geometric_tensor(subspace=subspace, i=i, j=j, delta=delta).real
    
    def berry_curvature(self, subspace=[0], i=0, j=0, delta=1e-4, recalculate=False):
        if not hasattr(self,'_dPks'):
            self.set_dPks(delta=delta)
        
        if recalculate:
            Pks = 0
            dPks_i = 0
            dPks_j = 0
            for n in subspace:
                # The band index is always in this position
                Pks = self._tb.Pks[...,n,:,:]
                dPks_i += self._dPks[i][...,n,:,:]
                dPks_j += self._dPks[j][...,n,:,:]
                
            self.check_projector(Pks, dPks_i, i)
            self.check_projector(Pks, dPks_j, j)

            omega = 1j * np.trace(Pks @ (dPks_i @ dPks_j - dPks_j @ dPks_i), axis1=-2, axis2=-1)
            if np.linalg.norm(omega.imag) > 1e-10:
                print('WARNING: The berry curvature has non-neglibile imaginary parts !')
            return omega.real
        else:
            return -2.0 * self.geometric_tensor(subspace=subspace, i=i, j=j, delta=delta).imag
        
    def chern_number(self, dim=2, subspace=[0], delta=1e-4):
        # FIXME what if not 2d?
        integrator = integrate.simpson
        if dim == 2:
            omega = self.berry_curvature(subspace=subspace, i=0, j=1, delta=delta)
            kx = self._kspace.cart_mesh[0][0]
            ky = self._kspace.cart_mesh[1][:,0]
            omega_int = integrator(integrator(omega, kx), ky)
            return (1.0 / (2.0 * np.pi)) * omega_int
    
    def metric_number(self, subspace=[0], i=0, j=0, delta=1e-4):
        # FIXME what if not 2D?
        integrator = integrate.simpson
        g = self.quantum_metric(subspace=subspace, i=i, j=j, delta=delta)
        kx = self._kspace.cart_mesh[0][0]
        ky = self._kspace.cart_mesh[1][:,0]
        g_int = integrator(integrator(g, kx), ky)
        return (1.0 / (2.0 * np.pi)) * g_int

    def plot_contour(self, subspace=[0], i=0, j=0, delta=1e-4,
                     cmap=cmap_cm.coolwarm, xlim=None, ylim=None):
        g = self.quantum_metric(subspace=subspace, i=i, j=j, delta=delta)
        omega = self.berry_curvature(subspace=subspace, i=i, j=j, delta=delta)

        levels = np.arange(0,5+0.1)
        levels = None

        fig, axis = plt.subplots(2)
        contour0 = axis[0].contourf(*self._kspace.cart_mesh/np.pi, g, cmap=cmap, levels=levels, extend='both')
        contour1 = axis[1].contourf(*self._kspace.cart_mesh/np.pi, omega, cmap=cmap, levels=levels, extend='both')

        for ax in axis:
            ax.set_xlabel('$k_x/\pi$')
            ax.set_ylabel('$k_y/\pi$')
            if xlim is not None:
                ax.set_xlim(np.array(xlim)/np.pi)
            if ylim is not None:
                ax.set_ylim(np.array(ylim)/np.pi)
        fig.colorbar(contour0)
        fig.colorbar(contour1)
        plt.show()
        
def hypercubic_kin(kspace_class, t=1, orb_dim=1):
    ks = kspace_class.cart_mesh
    di = np.diag_indices(orb_dim)
    Hks = np.zeros(shape=(*ks[0].shape, orb_dim, orb_dim))
    Hks[...,di[0],di[1]] = 2.0 * t * np.sum(np.cos(ks), axis=0)[..., None]
    return Hks

chain_lattice_vectors = [[1]]
chain_kspace = KSpace(lattice_vectors=chain_lattice_vectors)
chain_kspace.monkhorst_pack()
chain = TightBinding(Hks_fnc=hypercubic_kin, kspace_class=chain_kspace)

square_lattice_vectors = [[1,0],[0,1]]
square_kspace = KSpace(lattice_vectors=square_lattice_vectors)
square_kspace.monkhorst_pack(nk_list=300)
square = TightBinding(Hks_fnc=hypercubic_kin, kspace_class=square_kspace)
#square.plot_contour()
#square.plot_surface()

cubic_lattice_vectors = [[1,0,0],[0,1,0],[0,0,1]]
cubic_kspace = KSpace(lattice_vectors=cubic_lattice_vectors)
cubic_kspace.monkhorst_pack()
cubic = TightBinding(Hks_fnc=hypercubic_kin, kspace_class=cubic_kspace)

#square_lattice_vectors = [[1,0],[0,1]]
#square_kspace = KSpace(lattice_vectors=square_lattice_vectors)
#square_high_symmetry_points = {'G':[0,0], 'X':[1,0], 'M':[1,1], 'G2':[0,0]}
#k_path = square_kspace.path(k_points=list(square_high_symmetry_points.values()))
#square = TightBinding(Hks_fnc=cubic_kin, kspace_class=square_kspace)
#square.plot()

def honeycomb_kin(kspace_class, t=1, Delta=0.2, t2=0.1):
    a1 = kspace_class.lattice_vectors[0]
    a2 = kspace_class.lattice_vectors[1]
    a3 = -(a1+a2)
    
    n1 = (a2 - a1) / 3.0
    n2 = (a3 - a2) / 3.0
    n3 = (a1 - a3) / 3.0
    ns = [n1, n2, n3]
    gammaks = 0
    for n in ns:
        gammaks += np.exp( 1j * kspace_class.cart_list @ n )

    m1 = n2 - n3
    m2 = n3 - n2
    m3 = n1 - n3
    ms = [m1, m2, m3]
    alphaks = 0
    for m in ms:
        alphaks += np.sin(kspace_class.cart_list @ m )
    
    orb_dim = 2
    Hks = np.zeros(shape=(kspace_class.nks, orb_dim, orb_dim), dtype=np.complex_)
    Hks[...,0,0] = Delta + 2*t2*alphaks
    Hks[...,1,1] = -Delta - 2*t2*alphaks
    Hks[...,0,1] = t * gammaks
    Hks[...,1,0] = t * gammaks.conj()
    return kspace_class.to_mesh(Hks, A_type='array')

honeycomb_lattice_vectors = [[-1.0/2.0, np.sqrt(3)/2.0], [-1.0/2.0, -np.sqrt(3)/2.0]]
#honeycomb_lattice_vectors = [[np.sqrt(3)/2.0, 1/2.0], [-np.sqrt(3)/2.0, 1/2.0]]
honeycomb_kspace = KSpace(lattice_vectors=honeycomb_lattice_vectors)
#honeycomb_kspace.monkhorst_pack(nk_list=300, domain=[-1,1])
honeycomb_kspace.monkhorst_pack(nk_list=300, basis='cartesian', domain=[-2*np.pi,2*np.pi])
honeycomb = TightBinding(Hks_fnc=honeycomb_kin, kspace_class=honeycomb_kspace)
honeycomb.plot_contour()
honeycomb.plot_surface()
honeycomb_top = Topology(tb=honeycomb)
honeycomb_top.plot_contour(i=0, j=1, subspace=[0])

honeycomb_frac_kspace = KSpace(lattice_vectors=honeycomb_lattice_vectors)
honeycomb_frac_kspace.monkhorst_pack(nk_list=1000, endpoint=False)
honeycomb_frac = TightBinding(Hks_fnc=honeycomb_kin, kspace_class=honeycomb_frac_kspace)
honeycomb_frac.plot_contour()
honeycomb_frac.plot_surface()
honeycomb_frac_top = Topology(tb=honeycomb_frac)
print(f'honeycomb Chern number = {honeycomb_frac_top.chern_number(subspace=[0])}')

#hexagonal_high_symmetry_points = {'G':[0,0], 'M':[1.0/2.0,0], 'K':[1.0/3.0,1.0/3.0], 'G2':[0,0]}
#honeycomb = KSpace(lattice_vectors=honeycomb_lattice_vectors)
#k_path = honeycomb.path(k_points=list(honeycomb_high_symmetry_points.values()))
#honeycomb.run(Hks_fnc=honeycomb_kin, mesh=k_path)
#honeycomb.plot()

def kagome_kin(kspace_class, t=1):
    a1 = kspace_class.lattice_vectors[0]
    a2 = kspace_class.lattice_vectors[1]
    a3 = -(a1+a2)

    orb_dim = 3
    Hks = np.zeros(shape=(kspace_class.nks, orb_dim, orb_dim), dtype=np.complex_)
    Hks[...,0,1] = 2.0 * t * np.cos( 0.5 * kspace_class.cart_list @ a2 )
    Hks[...,0,2] = 2.0 * t * np.cos( 0.5 * kspace_class.cart_list @ a1 )
    Hks[...,1,2] = 2.0 * t * np.cos( 0.5 * kspace_class.cart_list @ a3 )
    Hks[...,1,0] = Hks[...,0,1].conj()
    Hks[...,2,0] = Hks[...,0,2].conj()
    Hks[...,2,1] = Hks[...,1,2].conj()
    return kspace_class.to_mesh(Hks, A_type='array')

kagome_lattice_vectors = [[-1.0/2.0, np.sqrt(3)/2.0], [-1.0/2.0, -np.sqrt(3)/2.0]]
kagome_kspace = KSpace(lattice_vectors=kagome_lattice_vectors)
kagome_kspace.monkhorst_pack(nk_list=300, basis='cartesian', domain=[-2*np.pi,2*np.pi])
kagome = TightBinding(Hks_fnc=kagome_kin, kspace_class=kagome_kspace)
#kagome.plot_contour(band=1)
#kagome.plot_surface()
kagome_top = Topology(tb=kagome)
#kagome_top.quantum_metric(subspace=[1])
#kagome_top.plot_contour(i=0, j=0, subspace=[1])


def lieb_kin(kspace_class, t=1, delta=0.3):
    a1 = kspace_class.lattice_vectors[0]
    a2 = kspace_class.lattice_vectors[1]

    orb_dim = 3
    Hks = np.zeros(shape=(kspace_class.nks, orb_dim, orb_dim), dtype=np.complex_)
    Hks[...,0,1] = 2.0 * t * ( np.cos( 0.5 * kspace_class.cart_list @ a1 ) 
                              + 1j * delta * np.sin( 0.5 * kspace_class.cart_list @ a1) )
    Hks[...,1,2] = 2.0 * t * ( np.cos( 0.5 * kspace_class.cart_list @ a2 )
                              + 1j * delta * np.sin( 0.5 * kspace_class.cart_list @ a2) )
    Hks[...,1,0] = Hks[...,0,1].conj()
    Hks[...,2,1] = Hks[...,1,2].conj()
    return kspace_class.to_mesh(Hks, A_type='array')

lieb_lattice_vectors = [[1,0],[0,1]]
lieb_kspace = KSpace(lattice_vectors=lieb_lattice_vectors)
lieb_kspace.monkhorst_pack(nk_list=300)
lieb = TightBinding(Hks_fnc=lieb_kin, kspace_class=lieb_kspace)
lieb.plot_contour(band=0)
lieb.plot_surface()
lieb_top = Topology(tb=lieb)
lieb_top.plot_contour(i=0, j=0, subspace=[1])