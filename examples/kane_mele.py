import numpy as np

from band_topology.meshes.kspace import KSpace
from band_topology.models.tightbinding import TightBinding
from band_topology.topology import Topology

# FIXME add in SOC term
    
def h_k(kspace_class, t=1, Lambda=0, Delta=0):
    # Identity in spin space, ham is orb \otimes spin dof.
    s0 = np.eye(2)
 
    def kay(i):
        return kspace_class.klist('crystal')[:,i]

    def ff(phi):
        return np.cos(kay(0) + phi) + np.cos(kay(1) - phi) + np.cos(kay(1) - kay(0) + phi)

    # Dispersion in crystal coordinates (2pi fractional coordinates)    
    Qks = np.exp(-1j * (kay(0) + kay(1)) / 3.0) \
        + np.exp(1j * (2*kay(0) - kay(1)) / 3.0) \
        + np.exp(1j * (2*kay(1) - kay(0)) / 3.0)

    orb_dim = 2
    Hks_up = np.zeros(shape=(kspace_class.nks, orb_dim, orb_dim), dtype=np.complex_)
    Hks_up[...,0,0] = Delta + 2 * Lambda * ff(phi=0.5*np.pi)
    Hks_up[...,1,1] = -Delta + 2 * Lambda * ff(phi=-0.5*np.pi)
    Hks_up[...,0,1] = t * Qks
    Hks_up[...,1,0] = t * Qks.conj()
    #Hks = np.kron(Hks, s0)
    
    Hks_dn = np.zeros(shape=(kspace_class.nks, orb_dim, orb_dim), dtype=np.complex_)
    Hks_dn[...,0,0] = Delta + 2 * Lambda * ff(phi=-0.5*np.pi)
    Hks_dn[...,1,1] = -Delta + 2 * Lambda * ff(phi=0.5*np.pi)
    Hks_dn[...,0,1] = t * Qks
    Hks_dn[...,1,0] = t * Qks.conj()

    Hks = np.zeros(shape=(kspace_class.nks, 2*orb_dim, 2*orb_dim), dtype=np.complex_)
    Hks[...,:2,:2] = Hks_up
    Hks[...,2:,2:] = Hks_dn
    return kspace_class.to_mesh(Hks, A_type='array')

# Define the hexagonal Bravais lattice
a1 = np.array( [np.sqrt(3)/2.0, -1.0/2.0] )
a2 = np.array( [np.sqrt(3)/2, 1.0/2.0] )
lattice_vectors = [a1, a2]
    
# Position of lattice sites
qa = (a1 + a2) / 3.0
qb = 2.0 * (a1 + a2) / 3.0

# The tb parameters
tb_parameters = dict(Lambda=1, Delta=1)

# Redefine on a path to get the spaghetti plots
# The path uses a dictionary that has to have unique keys, so we use the @ char to distinguish them, and all
# @ get removed for printing/plotting
high_symmetry_points = {'$\Gamma$':[0,0], '$M$':[1.0/2.0,0], '$K$':[2.0/3.0,1.0/3.0], '$\Gamma$@':[0,0]}
kspace = KSpace(lattice_vectors=lattice_vectors)
kspace.path(special_points=high_symmetry_points)
tb = TightBinding(Hks_fnc=h_k, kspace_class=kspace, tb_parameters=tb_parameters)
tb.plot_path()

# Redefine on a mesh in fractional coordinates corresponding to the a single BZ
frac_kspace = KSpace(lattice_vectors=lattice_vectors)
frac_kspace.monkhorst_pack(nk_list=60)
frac_tb = TightBinding(Hks_fnc=h_k, kspace_class=frac_kspace, tb_parameters=tb_parameters)
frac_tb.plot_contour()
frac_tb.plot_surface()

# Calculate the topology properties in the bottom band
subspace = [0,1]
top = Topology(tb=frac_tb, subspace=subspace, delta=1e-4)
print(f'Chern number = {top.chern_number()}')
print(f'metric number = {top.metric_number()}')

# Plot the xy elements of the geometric tensor
#top.plot_contour(function=top.quantum_metric, label='$g$')
#top.plot_contour(function=top.berry_curvature, label='$\Omega$')
#top.plot_colormesh(function=top.quantum_metric, label='$g$')
#top.plot_colormesh(function=top.berry_curvature, label='$\Omega$')

q = [qa, qb]
V = np.zeros([len(q),len(q)], dtype=np.complex_)
for i in range(len(q)):
    G = top._kspace.reciprocal_vectors[0]
    V[i,i] = np.exp(1j * G @ q[i])
V = np.kron(V, np.eye(2)) # Add spin dof

# Calculate Wilson loops
k2 = np.linspace(-0.5, 0.5, 100)
k2 = np.linspace(0, 1, 100)

W = np.empty(shape=(len(k2), len(subspace), len(subspace)), dtype=np.complex_)
for i,k in enumerate(k2):
    W[i,...] = top.wilson_path(n_points=100, path={'-pi':[-0.5,k], 'pi':[0.5,k]}, V=V)

e, v = np.linalg.eig(W)
wannier_centers = np.sort(np.angle(e), axis=-1) / (2*np.pi)
#wannier_centers = np.sort(-np.log(e).imag, axis=-1) / (2*np.pi)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(k2, 2*wannier_centers, 'o', color='black')
ax.set_ylim([-1, 1])
ax.set_xlim([min(k2), max(k2)])
ax.set_xlabel(r'$k_2 / \pi$')
ax.set_ylabel(r'$\mathcal{v} / \pi$')
plt.show()
