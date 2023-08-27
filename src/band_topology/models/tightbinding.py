import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm as cmap_cm
import seaborn as sns

class TightBinding():
    def __init__(self, Hks_fnc, kspace_class, tb_parameters=dict()):
        self._Hks_fnc = Hks_fnc
        self._kspace = kspace_class
        self._tb_parameters = tb_parameters
        self.set_Hks()

    def set_Hks(self):
        self._Hks = self._Hks_fnc(self._kspace, **self._tb_parameters)
        self._Eks, self._Uks = np.linalg.eigh(self._Hks)
        self._nbands = self._Eks.shape[-1]
        #return self._Hks

    def set_Pks(self):
        if not hasattr(self,'_Uks'):
            self.get_Hks()

        #v1 = self.Uks[0,0][:,0][:,np.newaxis] # make into a column vector, but not needed for outer
        # Several ways to do the same thing
        #P1 = v1 @ v1.conj().T
        #P1 = np.outer(v1, v1.conj())
        #P1 = np.einsum('i,j->ij', v1.ravel(), v1.conj().ravel())
        #P1 = np.multiply.outer(v1.ravel(), v1.conj().ravel())
        #P1 = np.tensordot(v1.ravel(), v1.conj().ravel(), axes=((), ()))
        #check = P1 @ v1 # should give back v1 if v1 is a column vector
        #err = np.linalg.norm(v1 - check)

        # maybe use https://pypi.org/project/opt-einsum/?
        self._Pks = np.einsum('...in,...jn->...nij', self.Uks, self.Uks.conj())
        # What is the equivalent tensordot operations?
        # Do I need np.moveaxis?
        # https://www.reddit.com/r/Python/comments/qevahz/tensor_contractions_with_numpys_einsum_function/
        #self._Pks = np.tensordot(self.Uks, self.Uks.conj(), axis=((-2),(-2)))

    def get_Uks_subspace(self, subspace):
        # nbands x len(subspace) rectangular matrix of eigenvectors, eg., U = [U1, U2, ..., UN]
        U = np.empty(shape=(*self._kspace.mesh_shape, self.nbands, len(subspace)), dtype=np.complex_)
        for n in range(len(subspace)):
            U[...,n] = self.Uks[...,subspace[n]]
        #return np.einsum('...in,...jn->...ij', U, U.conj())
        return U

    def get_Pks(self, subspace, U=None, method='vector'):
        # Projectors is P = U U^+
        if method == 'vector':
            if U is None:
                U = self.get_Uks_subspace(subspace)
            return U @ np.swapaxes(U.conj(), -2, -1)
        elif method == 'sum':
            Pks = 0
            for n in subspace:
                Pks += self.Pks[...,n,:,:]
            return Pks
        else:
            raise ValueError('Unrecognized method !')
    
    #FIXME
    def plot_path(self, ax=None):
        fig, ax = plt.subplots()
        
        x = [i for i in range(self._kspace.nks)]
        for n in range(self.nbands):
            ax.plot(x, self._Eks[...,n], color='black')

        xticks = list( self._kspace._path_class.special_points_indices.values() )
        xlabels = list( self._kspace._path_class.special_points_indices.keys() )
        for i in range(len(xlabels)):
            xlabels[i] = xlabels[i].replace('@', '')
        ax.set_xticks(xticks, xlabels)
        ax.set_xlim(xmin=0, xmax=len(x)-1)
        ax.set_ylabel(r'$\varepsilon(\vec{k})$')
        plt.show()

    def plot_contour(self, band=0, 
                           cmap=sns.color_palette('icefire', 
                           as_cmap=True),
                           levels=None,
                           basis='cartesian'):
        if basis == 'cartesian':
            ks = self._kspace.cart_mesh/np.pi
        elif basis == 'fractional':
            ks = self._kspace.frac_mesh
        
        fig, ax = plt.subplots()
        contour = ax.contourf(*ks, self.Eks[...,band], cmap=cmap)
        
        cb = fig.colorbar(contour, orientation='vertical', pad=0.01)
        cb.outline.set_visible(False)
        cb.ax.tick_params(width=0)
        cb.set_label(r'$\varepsilon(\vec{k})$')

        if basis == 'cartesian':
            ax.set_xlabel('$k_x/\pi$')
            ax.set_ylabel('$k_y/\pi$')
        elif basis == 'fractional':
            ax.set_xlabel('$k_1$')
            ax.set_ylabel('$k_2$')
        
        plt.show()
    
    def plot_surface(self, band=None, 
                           cmap=sns.color_palette('icefire', as_cmap=True), 
                           vmin=None, 
                           vmax=None,
                           basis='cartesian'):
        if basis == 'cartesian':
            ks = self._kspace.cart_mesh/np.pi
        elif basis == 'fractional':
            ks = self._kspace.frac_mesh
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        if band is None:
            if vmin is None:
                vmin = np.min(self.Eks)
                vmax = np.max(self.Eks)
            for n in range(self.nbands):
                surf = ax.plot_surface(*ks, self.Eks[...,n], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            surf = ax.plot_surface(*ks, self.Eks[...,band], cmap=cmap, vmin=vmin, vmax=vmax)
        
        #cb = fig.colorbar(surf, orientation='vertical', pad=0.01)
        #cb.outline.set_visible(False)
        #cb.ax.tick_params(width=0)
        #cb.set_label(r'$\varepsilon(\vec{k})$')
        
        ax.set_zlabel(r'$\varepsilon(\vec{k})$')
        
        if basis == 'cartesian':
            ax.set_xlabel('$k_x/\pi$')
            ax.set_ylabel('$k_y/\pi$')
        elif basis == 'fractional':
            ax.set_xlabel('$k_1$')
            ax.set_ylabel('$k_2$')
        
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
    def Uks(self):
        if not hasattr(self,'_Uks'):
            self.set_Hks()
        return self._Uks
    
    @property
    def Pks(self):
        if not hasattr(self,'_Pks'):
            self.set_Pks()
        return self._Pks    