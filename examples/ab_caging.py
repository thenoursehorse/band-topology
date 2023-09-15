import numpy as np

from band_topology.meshes.kspace import KSpace
from band_topology.models.tightbinding import TightBinding
from band_topology.topology import Topology

def h_k(kspace_class, J=-1, beta1=np.pi/4, beta2=np.pi/4, Delta_a=0, Delta_b=0, Delta_c=0, gamma_a=0, gamma_b=0, gamma_c=0):

    def kay(i):
        return kspace_class.klist('crystal')[:,i]
    
    orb_dim = 3
    Hks = np.zeros(shape=(kspace_class.nks, orb_dim, orb_dim), dtype=np.complex_)
    
    Hks[...,0,0] = Delta_a - 1j * gamma_a
    Hks[...,1,1] = Delta_b - 1j * gamma_b
    Hks[...,2,2] = Delta_c - 1j * gamma_c

    #Hks[...,0,1] = J * ( np.exp(1j * beta1) + np.exp(1j * kay(0)) * np.exp(-1j * beta2) )
    #Hks[...,0,2] = J * ( np.exp(-1j * beta2) + np.exp(1j * kay(0)) * np.exp(1j * beta1) )
    
    #Hks[...,0,1] = J * ( 1 + np.exp(1j * kay(0)) * np.exp(-1j * beta2) * np.exp(-1j * beta1))
    #Hks[...,0,2] = J * ( np.exp(-1j * beta2) * np.exp(-1j * beta2) + np.exp(1j * kay(0)) )

    Hks[...,0,1] = J * (1 + np.exp(1j * kay(0)) * np.exp(-1j * kay(1)))
    Hks[...,0,2] = J * (np.exp(-1j * kay(1)) + np.exp(1j * kay(0)))

    Hks[...,1,0] = Hks[...,0,1].conj()
    Hks[...,2,0] = Hks[...,0,2].conj()
    Hks[...,2,1] = Hks[...,1,2].conj()
    
    return kspace_class.to_mesh(Hks, A_type='array')

# Define the square Bravais lattice
lattice_vectors = [[1,0], [0,1]]
frac_high_symmetry_points = {'$\Gamma$':[0,0], '$X$':[0,1.0/2.0], '$M$':[1.0/2.0,1.0/2.0], '$\Gamma$@':[0,0]}
#frac_high_symmetry_points = {'$\Gamma$':[0,1/4.0], '$X$':[1,1/4.0]} # This is the AB cage parameters

# The parameters of the model
tb_parameters = dict(J=-1, Delta_b=0.5, Delta_c=-0.5)

# Define the meshgrid and tight-binding model
kspace = KSpace(lattice_vectors=lattice_vectors)
kspace.monkhorst_pack(nk_list=100)
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

# Plot the xy elements of the geometric tensor
top.plot_contour(function=top.quantum_metric, label='$g$')
top.plot_contour(function=top.berry_curvature, label='$\Omega$')

# Calculate Wilson loops
k2 = np.linspace(0, 1, 100)
W2 = np.empty(shape=(len(k2), len(subspace), len(subspace)), dtype=np.complex_)
for i,k in enumerate(k2):
    W2[i,...] = top.wilson_path(n_points=100, path={'0':[0,k], '2pi':[1,k]})
e2, v2 = np.linalg.eig(W2)
wannier_centers_2 = np.sort(np.angle(e2), axis=-1) / (2*np.pi)

k1 = np.linspace(0, 1, 100)
W1 = np.empty(shape=(len(k1), len(subspace), len(subspace)), dtype=np.complex_)
for i,k in enumerate(k1):
    W1[i,...] = top.wilson_path(n_points=100, path={'0':[k,0], '2pi':[k,1]})
e1, v1 = np.linalg.eig(W1)
wannier_centers_1 = np.sort(-np.log(e1).imag, axis=-1) / (2*np.pi)

import matplotlib.pyplot as plt
fig, axis = plt.subplots(1,2)
axis[0].plot(2*k2, 2*wannier_centers_2, '.', color='black')
axis[0].set_ylim([-1, 1])
axis[0].set_xlim([2*min(k2), 2*max(k2)])
axis[0].set_xlabel(r'$k_2 / \pi$')
axis[0].set_ylabel(r'$\phi_x / \pi$')

axis[1].plot(2*k1, 2*wannier_centers_1, '.', color='black')
axis[1].set_ylim([-1, 1])
axis[1].set_xlim([2*min(k1), 2*max(k1)])
axis[1].set_xlabel(r'$k_1 / \pi$')
axis[1].set_ylabel(r'$\phi_y / \pi$')
plt.show()