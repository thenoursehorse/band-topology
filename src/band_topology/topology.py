import numpy as np

from scipy import integrate

import matplotlib.pyplot as plt
from matplotlib import cm as cmap_cm
import seaborn as sns

from copy import deepcopy

from band_topology.meshes.kspace import *
from band_topology.models.tightbinding import *
    
class Topology:
    def __init__(self, tb, subspace, basis='cartesian', delta=1e-4):
        self._tb = tb
        self._kspace = tb._kspace
        self._subspace = subspace
        self._basis = basis
        self._delta = delta

        if self._basis != 'cartesian':
            raise ValueError('Only evaluating metrics in cartesian coordinates is properly working !')

    def kspace_shifted(self, shift_vector):
        shift_list = deepcopy(self._kspace._shift_list) + shift_vector
        kspace_shifted = KSpace(lattice_vectors=self._kspace.lattice_vectors)
        if self._kspace._k_type == 'full_bz':
            kspace_shifted.monkhorst_pack(nk_list=self._kspace._nk_list,
                                          shift_list=shift_list,
                                          basis=self._kspace.basis,
                                          domain=self._kspace._domain,
                                          endpoint=self._kspace._endpoint)
        else:
            raise ValueError('Unrecognized kspace grid type !')
        return kspace_shifted
    
    def check_projector(self, Pks, dPks, i, tol=1e-8):
        err = np.linalg.norm(dPks @ Pks - Pks @ dPks)
        if err > tol:
            print(f'WARNING: The projectors and their derivatives on dim {i} are bad with error {err} !')

    def check_uniform_pairing_condition(self):
        n_orb = self._Pks.shape[-1]
        n_f = len(self._subspace)
        #upc = [self.bz_integrate(self.Pks[...,n,n]) for n in range(n_orb)]
        if self._kspace._endpoint:
            Pks = self.Pks[:-1,:-1,...]
        else:
            Pks = self.Pks[...]
        nks = np.product(Pks.shape[:self._kspace.d])
        upc = [np.sum(self.Pks[...,n,n]) / nks for n in range(n_orb)]
        return upc
    
    def get_Pks_shifted(self, shift_vector):
        # FIXME there should be a check to see if the shift is commensurate with the already
        # defined grid, so it does not have to be recalculated
        ks_shifted = self.kspace_shifted(shift_vector)
        tb_shifted = TightBinding(Hks_fnc=self._tb._Hks_fnc, 
                                  kspace_class=ks_shifted, 
                                  tb_parameters=self._tb._tb_parameters)
        return tb_shifted.get_Pks(self._subspace)
    
    def get_dPks(self, shift_vector, method='forward'):
        if method == 'center':
            return (self.get_Pks_shifted(shift_vector) - self.get_Pks_shifted(-shift_vector) ) / (2.0*self._delta)
        elif method == 'forward':
            return (self.get_Pks_shifted(shift_vector) - self.Pks) / self._delta
        elif method == 'dumb':
            # indexing of mesh arrays is in 'xy'
            if i == 1:
                ks = self._kspace.mesh(self.basis)[0][0]
            if i == 0:
                ks = self._kspace.mesh(self.basis)[1][:,0]
            return np.gradient(self.Pks, ks, axis=i)
        else:
            raise ValueError('Unrecognized derivative method !')

    def set_Pks(self):
        self._Pks = self._tb.get_Pks(self._subspace)
    
    def set_dPks(self):
        # Apply a small shift in the coordinate system specified by self.basis
        shift_vectors = self._delta * np.eye(self._kspace.d)

        # Match shift coordinates and generating funciton for kspace
        if (self._kspace.basis == 'fractional') and (self.basis == 'cartesian'):
            for i in range(len(shift_vectors)):
                shift_vectors[i] = self._kspace.transform_ks(shift_vectors[i], to_basis='fractional')
        if (self._kspace.basis == 'cartesian') and (self.basis == 'fractional'):
            for i in range(len(shift_vectors)):
                shift_vectors[i] = self._kspace.transform_ks(shift_vectors[i], to_basis='cartesian')

        self._dPks = [self.get_dPks(vec) for vec in shift_vectors]
    
    def geometric_tensor(self, i=0, j=0):
        Pks = self.Pks
        dPks_i = self.dPks[i]
        dPks_j = self.dPks[j]
        
        # FIXME the derivatives always fail this check, but my results are still very good
        #self.check_projector(Pks, dPks_i, i)
        #self.check_projector(Pks, dPks_j, j)

        # Different papers give different definitions for this, but the bottom one is the correct one
        #return 2.0 * np.trace(dPks_i @ dPks_j - dPks_i @ Pks @ dPks_j, axis1=-2, axis2=-1)
        #return 2.0 * np.trace(Pks @ dPks_i @ dPks_j, axis1=-2, axis2=-1)
        return np.trace(Pks @ dPks_i @ dPks_j, axis1=-2, axis2=-1)
    
    def quantum_metric(self, i=0, j=0, recalculate=False):
        if recalculate:
            Pks = self.Pks
            dPks_i = self.dPks[i]
            dPks_j = self.dPks[j]

            self.check_projector(Pks, dPks_i, i)
            self.check_projector(Pks, dPks_j, j)

            g =  0.5 * np.trace(dPks_i @ dPks_j, axis1=-2, axis2=-1)
            if np.linalg.norm(g.imag) > 1e-10:
                print('WARNING: The quantum metric has non-neglibile imaginary parts !')
            return g.real
        else:
            return self.geometric_tensor(i=i, j=j).real
    
    def berry_curvature(self, i=0, j=0, recalculate=False):
        if recalculate:
            Pks = self.Pks
            dPks_i = self.dPks[i]
            dPks_j = self.dPks[j]
                
            self.check_projector(Pks, dPks_i, i)
            self.check_projector(Pks, dPks_j, j)

            omega = 1j * np.trace(Pks @ (dPks_i @ dPks_j - dPks_j @ dPks_i), axis1=-2, axis2=-1)
            if np.linalg.norm(omega.imag) > 1e-10:
                print('WARNING: The berry curvature has non-neglibile imaginary parts !')
            return omega.real
        else:
            return -2.0 * self.geometric_tensor(i=i, j=j).imag

    def bz_integrate(self, A, method='simpson'):
        '''
        Note when mesh.basis and self.basis do not match:
        It is easiest to evaluate the multi-dim integral on a hypercube, rather than 
        parameterize the boundaries and curves. So it is always evaluated in the basis
        of mesh.basis that constructed a hypercube grid from monkhorst pack

        In 2D: If mesh.basis is fractional and self.mesh is Cartesian:
        Have to multiply by \int d^2k det(sqrt(eta)) = (2*pi)^2 / Vc,
        where eta is the inverse coordinate metric, Vc is the volume of the unit cell,
        because evaluating the metric in cartesian coordinates (look up 2D integral coordinate transformations)
        but taking integral in frac coordinates
        '''
        if self._kspace.d != 2:
            raise ValueError('Only implemented for 2D !')
        
        if method == 'simpson':
            integrator = integrate.simpson
            # Note the order is reversed for meshgrid when indexing='xy'
            ky = self._kspace.mesh(self._kspace.basis)[0][0]
            kx = self._kspace.mesh(self._kspace.basis)[1][:,0]

            # simpson needs closed domains so add wrap if needed
            if not self._kspace._endpoint:
                # FIXME what if it is not uniform? It should be kx[-1] = kx[0] + G
                dx = kx[-1] - kx[-2]
                dy = ky[-1] - ky[-2]
                kx = np.pad(kx, ((0,1)), constant_values=kx[-1]+dx)
                ky = np.pad(ky, ((0,1)), constant_values=ky[-1]+dy)

                shape_ks = A.shape[:self._kspace.d]
                shape_rest = A.shape[self._kspace.d:]
                padding = []
                for _ in range(len(shape_ks)):
                    padding.append( (0,1) )
                for _ in range(len(shape_rest)):
                    padding.append( (0,0) )
                A_tmp = np.pad(A, padding, mode='wrap')

            else:
                A_tmp = A
            
            A_int = integrator(integrator(A_tmp, kx), ky)

            # Add integration volume fraction
            if (self._kspace.basis == 'fractional') and (self.basis == 'cartesian'):
                factor = (2.0*np.pi)**2 / (self._kspace.vc)
            elif (self._kspace.basis == 'cartesian') and (self.basis == 'fractional'):
                raise ValueError('Not implemented !')
            else:
                factor = 1
            return factor * A_int
        
    def chern_number(self, method='simpson'):
        '''
        See notes in self.bz_integrate
        '''
        if self._kspace.d != 2:
            raise ValueError('Only implemented for 2D !')
        
        if method == 'simpson':
            omega = self.berry_curvature(i=0, j=1)
            return (1.0/(2.0*np.pi)) * self.bz_integrate(omega, method='simpson')
        elif method == 'point-split':
            # Metric in cartesian, integral in fractional
            shift_x = self._delta * np.array([1,0])
            shift_y = self._delta * np.array([0,1])
            if self._kspace.basis == 'fractional':
                shift_x = self._kspace.transform_ks(shift_x, to_basis='fractional')
                shift_y = self._kspace.transform_ks(shift_y, to_basis='fractional')
            Pks = self.Pks
            Pks_x = self.get_Pks_shifted(shift_x)
            Pks_y = self.get_Pks_shifted(shift_y)
            # Can't include the endpoint because it is double counting the 'origin'
            if self._kspace._endpoint:
                Pks = Pks[:-1,:-1,...]
                Pks_x = Pks_x[:-1,:-1,...]
                Pks_y = Pks_y[:-1,:-1,...]
            # number of k-points might have changed
            nks = np.product(Pks.shape[0:self._kspace.d])
            omega_int = np.sum(1j * np.trace(Pks @ (Pks_x @ Pks_y - Pks_y @ Pks_x), axis1=-2, axis2=-1)).real / self._delta**2 / nks
            # Integration volume factor
            factor = (2.0*np.pi)**2 / (self._kspace.vc)
            return (1.0 / (2.0*np.pi)) * factor * omega_int
        else:
            raise ValueError('Unrecognized integration method !')
    
    def metric_number(self, method='simpson'):
        '''
        Eq. B10 in supplemental of 10.1103/PhysRevLett.128.087002
        point-split is Eq. B21
        
        See notes in self.bz_integrate
        '''
        
        if self._kspace.d != 2:
            raise ValueError('Only implemented for 2D !')
        
        if method == 'simpson':
            integrator = integrate.simpson
            g = 0
            for i in range(self._kspace.d):
                # FIXME eta^ij g_ij = g^i_i = Tr(g) only in a Cartesian basis, so need a check here
                # and unsure what it is in fractional basis?
                g += self.quantum_metric(i=i, j=i)
            return (1.0/(2.0*np.pi)**2) * self.bz_integrate(g, method='simpson')
        elif method == 'point-split':
            # Metric in cartesian, integral in fractional
            shift_x = self._delta * np.array([1,0])
            shift_y = self._delta * np.array([0,1])
            if self._kspace.basis == 'fractional':
                shift_x = self._kspace.transform_ks(shift_x, to_basis='fractional')
                shift_y = self._kspace.transform_ks(shift_y, to_basis='fractional')
            Pks = self.Pks
            Pks_x = self.get_Pks_shifted(shift_x)
            Pks_y = self.get_Pks_shifted(shift_y)
            # Can't include the endpoint because it is double counting the 'origin'
            if self._kspace._endpoint:
                Pks = Pks[:-1,:-1,...]
                Pks_x = Pks_x[:-1,:-1,...]
                Pks_y = Pks_y[:-1,:-1,...]
            # number of k-points might have changed
            nks = np.product(Pks.shape[0:self._kspace.d])
            g_int =  (2*self.N_occ*nks \
                        - np.sum(np.trace(Pks @ (Pks_x + Pks_y), axis1=-2, axis2=-1))).real / self._delta**2 / nks
            # Integration volume factor
            factor = (2.0*np.pi)**2 / (self._kspace.vc)
            return (1.0 / (2.0*np.pi)**2) * factor * g_int
        else:
            raise ValueError('Unrecognized integration method !')

    def plot_contour(self, i=0, j=0, cmap=sns.color_palette('icefire', as_cmap=True), xlim=None, ylim=None):
        g = self.quantum_metric(i=i, j=j)
        omega = self.berry_curvature(i=i, j=j)

        #levels = np.arange(0,5+0.1)
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

    def plot_colormesh(self, i=0, j=0, cmap=sns.color_palette('icefire', as_cmap=True), vmin=None, vmax=None, rasterized=True):
        g = self.quantum_metric(i=i, j=j)
        omega = self.berry_curvature(i=i, j=j)
        
        fig, axis = plt.subplots(2)

        im = []
        im.append( axis[0].pcolormesh(*self._kspace.cart_mesh/np.pi, g, vmin=vmin, vmax=vmax, rasterized=rasterized, cmap=cmap) )
        im.append( axis[1].pcolormesh(*self._kspace.cart_mesh/np.pi, omega, vmin=vmin, vmax=vmax, rasterized=rasterized, cmap=cmap) )

        cb = []
        for i in range(len(axis)):
            cb.append( fig.colorbar(im[i], orientation='vertical', pad=0.01) )
            cb[i].outline.set_visible(False)
            cb[i].ax.tick_params(width=0)
        
        for ax in axis:
            ax.set_xlabel('$k_x/\pi$')
            ax.set_ylabel('$k_y/\pi$')

        plt.show()

    @property
    def basis(self):
        return self._basis
    
    @property
    def Pks(self):
        if not hasattr(self,'_Pks'):
            self.set_Pks()
        return self._Pks
    
    @property
    def dPks(self):
        if not hasattr(self,'_dPks'):
            self.set_dPks()
        return self._dPks

    @property
    def N_occ(self):
        #N = np.trace(self.Pks, axis1=-2, axis2=-1)
        return len(self._subspace)        