import numpy as np

from band_topology.meshes.kspace import KSpace
from band_topology.models.tightbinding import TightBinding
from band_topology.topology import Topology

def Gamma(i,j):
    sigma = []
    sigma.append( np.eye(2, dtype=np.complex_) )
    sigma.append( np.array([[0,1],[1,0]], dtype=np.complex_) )
    sigma.append( np.array([[0,-1j],[1j,0]], dtype=np.complex_) )
    sigma.append( np.array([[1,0],[0,-1]], dtype=np.complex_) )
    return np.kron(sigma[i], sigma[j])

def h_k(kspace_class, alpha=0, beta=0, delta=0, mmm=0, model='eq17'):
    ks = kspace_class.cart_list
    def kay(i):
        return ks[:,i]
    
    if model == 'eq17':
        Hks = (1.0/2.0) * Gamma(1,3) \
              + (alpha/2.0) *  (Gamma(3,0) + Gamma(0,3)) * (np.cos(kay(0)) + np.cos(kay(1)))[...,None,None] \
              + (Gamma(1,2) + Gamma(3,1)) * np.sin(kay(0))[...,None,None] \
              + (Gamma(2,1) + Gamma(3,2)) * np.sin(kay(1))[...,None,None] \
              - Gamma(0,3) \
              + (beta/2.0) * (Gamma(0,3) - Gamma(3,0)) * (np.cos(kay(0)) * (np.cos(kay(1)) - delta))[...,None,None]
    elif model == 'eq20':
        Hks = (2.0 - mmm - np.cos(kay(0)) - np.cos(kay(1)))[...,None,None] * Gamma(0,3) \
              - delta * np.sin(kay(1))[...,None,None] * Gamma(1,2) \
              + np.sin(kay(0))[...,None,None] * (Gamma(3,1) + Gamma(1,1)) \
              + np.sin(kay(1))[...,None,None] * (Gamma(2,1) + Gamma(0,2))
    else:
        raise ValueError('Unrecognized model !')
    return kspace_class.to_mesh(Hks, A_type='array')

# Define the square Bravais lattice
lattice_vectors = [[1,0],[0,1]]
frac_high_symmetry_points = {'$\Gamma$':[0,0], '$X$':[0,1.0/2.0], '$M$':[1.0/2.0,1.0/2.0], '$\Gamma$@':[0,0]}

# The parameters for each model
#tb_parameters = dict(alpha=-1.5, beta=1.5, delta=1, model='eq17') # C=0
tb_parameters = dict(alpha=-1.5, beta=0.0, delta=1, model='eq17') # C=-1

#tb_parameters = dict(mmm=4.3, delta=0, model='eq20') # Trivial TRS Z2
#tb_parameters = dict(mmm=3.0, delta=0, model='eq20') # Topological TRS Z2


# Define the meshgrid and tight-binding model
kspace = KSpace(lattice_vectors=lattice_vectors)
kspace.monkhorst_pack(nk_list=60)
tb = TightBinding(Hks_fnc=h_k, kspace_class=kspace, tb_parameters=tb_parameters)
tb.plot_surface()

# Redefine on a path to get the spaghetti plots
path_kspace = KSpace(lattice_vectors=lattice_vectors)
path_kspace.path(special_points=frac_high_symmetry_points)
path_tb = TightBinding(Hks_fnc=h_k, kspace_class=path_kspace, tb_parameters=tb_parameters)
path_tb.plot_path()

# Calculate the topology properties in flat band in the middle
subspace=[0,1]
top = Topology(tb=tb, subspace=subspace)
print(f'Chern number = {top.chern_number()}')
print(f'metric number = {top.metric_number()}')

# Plot the xx elements of the geometric tensor
#top.plot_contour(function=top.quantum_metric, i=0, j=1, label='$g$') # to only plot a single element of the tensor
top.plot_contour(function=top.quantum_metric, label='$g$')
top.plot_colormesh(function=top.quantum_metric, label='$g$')
top.plot_contour(function=top.berry_curvature, label='$\Omega$')
top.plot_colormesh(function=top.berry_curvature, label='$\Omega$')

# Calculate Wilson loops
k2 = np.linspace(-0.5, 0.5, 100)
#k2 = np.linspace(-np.pi, np.pi, 100)

W = np.empty(shape=(len(k2), len(subspace), len(subspace)), dtype=np.complex_)
for i,k in enumerate(k2):
    W[i,...] = top.wilson_path(n_points=100, path={'-pi':[-0.5,k], 'pi':[0.5,k]})
    #W[i,...] = top.wilson_path(n_points=10000, path={'-pi':[k,-0.5], 'pi':[k,0.5]})
    #W[i,...] = top.wilson_path(n_points=10000, path={'-pi':[-np.pi,k], 'pi':[np.pi,k]}, basis='cartesian')

e, v = np.linalg.eig(W)
#wannier_centers = np.sort(np.angle(e), axis=-1) / (2*np.pi)
wannier_centers = np.sort(-np.log(e).imag, axis=-1) / (2*np.pi)

# Chern number should be
# velocity = np.gradient(wannier_centers, k2, axis=0)
# integrate velocity from 0->2pi in ky and add both bands for Chern number

import matplotlib.pyplot as plt
plt.plot(k2, 2*wannier_centers, 'o', color='black')
plt.show()