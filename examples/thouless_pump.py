import numpy as np

from band_topology.meshes.kspace import KSpace
from band_topology.models.tightbinding import TightBinding
from band_topology.topology import Topology

def pauli(i):
    if (i == 0) or (i =='id'):
        return np.eye(2, dtype=np.complex_)
    elif (i == 1) or (i == 'x'):
        return np.array([[0,1],[1,0]], dtype=np.complex_)
    elif (i == 2) or (i == 'y'):
        return np.array([[0,-1j],[1j,0]], dtype=np.complex_)
    elif (i == 3) or (i == 'z'):
        return np.array([[1,0],[0,-1]], dtype=np.complex_)
    else:
        raise ValueError('Unrecognized Pauli matrix !')

def h_k(kspace_class):
    def kay(i):
        return kspace_class.klist('crystal')[:,i]
    
    Hks = pauli('z') * np.ones(shape=kay(0).shape)[...,None,None] \
        + pauli('z') * (np.cos(kay(0)) + np.cos(kay(1)))[...,None,None] \
        + pauli('y') * np.sin(kay(0))[...,None,None] \
        + pauli('x') * np.sin(kay(1))[...,None,None]
    return kspace_class.to_mesh(Hks, A_type='array')

# Define the square Bravais lattice
lattice_vectors = [[1,0], [0,1]]
frac_high_symmetry_points = {'$\Gamma$':[0,0], '$X$':[0,1.0/2.0], '$M$':[1.0/2.0,1.0/2.0], '$\Gamma$@':[0,0]}

# Define the meshgrid and tight-binding model
kspace = KSpace(lattice_vectors=lattice_vectors)
kspace.monkhorst_pack(nk_list=60)
tb = TightBinding(Hks_fnc=h_k, kspace_class=kspace)
tb.plot_contour()
tb.plot_surface()

# Redefine on a path to get the spaghetti plots
path_kspace = KSpace(lattice_vectors=lattice_vectors)
path_kspace.path(special_points=frac_high_symmetry_points)
path_tb = TightBinding(Hks_fnc=h_k, kspace_class=path_kspace)
path_tb.plot_path()

# Calculate the topology properties in flat band in the middle
subspace=[0]
top = Topology(tb=tb, subspace=subspace)
print(f'Chern number = {top.chern_number()}')
print(f'metric number = {top.metric_number()}')

# Plot the xy elements of the geometric tensor
top.plot_contour(function=top.quantum_metric, label='$g$')
top.plot_contour(function=top.berry_curvature, label='$\Omega$')

# Calculate Wilson loops
k2 = np.linspace(-0.5, 0.5, 100)

W = np.empty(shape=(len(k2), len(subspace), len(subspace)), dtype=np.complex_)
for i,k in enumerate(k2):
    W[i,...] = top.wilson_path(n_points=100, path={'0':[0,k], '2pi':[1,k]})

e, v = np.linalg.eig(W)
wannier_centers = np.sort(np.angle(e), axis=-1) / (2*np.pi)

# Chern number should be
velocity = np.gradient(wannier_centers, k2, axis=0)
# integrate velocity from 0->2pi in ky and add both bands for Chern number

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(k2, 2*wannier_centers, 'o', color='black')
ax.set_ylim([-1, 1])
ax.set_xlim([min(k2), max(k2)])
ax.set_xlabel(r'$k_2 / \pi$')
ax.set_ylabel(r'$\mathcal{v} / \pi$')
plt.show()