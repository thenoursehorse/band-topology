import numpy as np

from band_topology.meshes.kspace import KSpace
from band_topology.models.tightbinding import TightBinding
from band_topology.topology import Topology

def get_M(ks, t=1, a=1):
    nks = ks.shape[0]
    
    M = np.zeros(shape=(nks, 6, 3), dtype=np.complex_)
    M[:,0,0] = np.exp(1j * (ks[:,0]*a - np.sqrt(3)*ks[:,1]*a)/4)
    M[:,0,2] = np.exp(1j * (-ks[:,0]*a + np.sqrt(3)*ks[:,1]*a)/4)
    M[:,1,0] = np.exp(1j * (-ks[:,0]*a + np.sqrt(3)*ks[:,1]*a)/4)
    M[:,1,2] = np.exp(1j * (ks[:,0]*a - np.sqrt(3)*ks[:,1]*a)/4)
    M[:,2,0] = np.exp(1j * (-ks[:,0]*a)/2)
    M[:,2,1] = np.exp(1j * (ks[:,0]*a)/2)
    M[:,3,0] = np.exp(1j * (ks[:,0]*a)/2)
    M[:,3,1] = np.exp(1j * (-ks[:,0]*a)/2)
    M[:,4,1] = np.exp(1j * (ks[:,0]*a + np.sqrt(3)*ks[:,1]*a)/4)
    M[:,4,2] = np.exp(1j * (-ks[:,0]*a - np.sqrt(3)*ks[:,1]*a)/4)
    M[:,5,1] = np.exp(1j * (-ks[:,0]*a - np.sqrt(3)*ks[:,1]*a)/4)
    M[:,5,2] = np.exp(1j * (ks[:,0]*a + np.sqrt(3)*ks[:,1]*a)/4)
    return -t * M
    
def lieb_kagome_kin(kspace_class, t=1, a=1, delta=0):
    a1 = kspace_class.lattice_vectors[0]
    a2 = kspace_class.lattice_vectors[1]

    ks = kspace_class.cart_list

    A = delta * np.array(np.eye(3), dtype=np.complex_)
    B = -delta * np.array(np.eye(6), dtype=np.complex_)
    M = get_M(ks, t, a)
    M_dag = M.conj().transpose((0,2,1))

    Hks = np.zeros(shape=(kspace_class.nks, 9, 9), dtype=np.complex_)
    Hks[:, :3, :3] = A
    Hks[:, 3:, 3:] = B
    Hks[:, :3, 3:] = M_dag
    Hks[:, 3:, :3] = M
    return kspace_class.to_mesh(Hks, A_type='array')

# Define the square Bravais lattice
a1 = np.array([2.0, 0.0])
a2 = np.array([-1.0, np.sqrt(3)])
lattice_vectors = [a1, a2]
frac_high_symmetry_points = {'$\Gamma$':[0,0], '$M$':[1.0/2.0,0], '$K$':[1.0/3.0,1.0/3.0], '$\Gamma$@':[0,0]}

# Position of lattice sites
qA = a1 - a1
qB = 0.5 * a1
qC = 0.5 * a2
qD = 0.25 * a2
qE = 0.75 * a2
qF = 0.25 * a1
qG = 0.75 * a1
qH = 0.75 * a2 + 0.25 * a1
qI = 0.25 * a2 + 0.75 * a1

# The parameters for the lieb lattice
tb_parameters = dict(delta=0) # 0.3

# Define the meshgrid and tight-binding model
kspace = KSpace(lattice_vectors=lattice_vectors)
kspace.monkhorst_pack(nk_list=60, domain=[-1*np.pi,1*np.pi], basis='cartesian')
tb = TightBinding(Hks_fnc=lieb_kagome_kin, kspace_class=kspace, tb_parameters=tb_parameters)
tb.plot_contour(band=-1)
tb.plot_surface()

# Redefine on a path to get the spaghetti plots
path_kspace = KSpace(lattice_vectors=lattice_vectors)
path_kspace.path(special_points=frac_high_symmetry_points)
path_tb = TightBinding(Hks_fnc=lieb_kagome_kin, kspace_class=path_kspace, tb_parameters=tb_parameters)
path_tb.plot_path()

# Redefine on a mesh in fractional coordinates corresponding to the a single BZ
frac_kspace = KSpace(lattice_vectors=lattice_vectors)
frac_kspace.monkhorst_pack(nk_list=60)
frac_tb = TightBinding(Hks_fnc=lieb_kagome_kin, kspace_class=frac_kspace, tb_parameters=tb_parameters)
frac_tb.plot_contour(band=-1, basis='fractional')
frac_tb.plot_surface(basis='fractional')

# Calculate the topology properties of the flat bands in the middle
subspace = [0,1,2,3,4,5]
top = Topology(tb=frac_tb, subspace=subspace)
print(f'Chern number = {top.chern_number()}')
print(f'metric number = {top.metric_number()}')

# Plot the xx elements of the geometric tensor
top.plot_contour(function=top.quantum_metric, label='$g$')
#top.plot_colormesh(function=top.quantum_metric, label='$g$')
top.plot_contour(function=top.berry_curvature, label='$\Omega$')
#top.plot_colormesh(function=top.berry_curvature, label='$\Omega$')

# FIXME work out V
# Is this the order Miriam put them in corresponding to h_k?
q = [qA, qB, qC, qD, qE, qF, qG, qH, qI]
V = np.zeros([len(q),len(q)], dtype=np.complex_)
for i in range(len(q)):
    G = top._kspace.reciprocal_vectors[0] # FIXME what should this be?
    V[i,i] = np.exp(1j * G @ q[i])

# Calculate Wilson loops
subspace = [0,1,2,3,4,5]
#subspace = [0,1,2]
#subspace = [3,4,5]
w_top = Topology(tb=frac_tb, subspace=subspace)
k2 = np.linspace(-0.5,0.5,100)
W = np.empty(shape=(len(k2), len(subspace), len(subspace)), dtype=np.complex_)
for i,k in enumerate(k2):
    W[i,...] = w_top.wilson_path(n_points=100, path={'0':[0,k], '2pi':[1,k]}, V=V)
    #W[i,...] = w_top.wilson_path(n_points=100, path={'0':[k,0], '2pi':[k,1]}, V=V)

e, v = np.linalg.eig(W)
wannier_centers = np.sort(np.angle(e), axis=-1) / (2*np.pi)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(k2, 2*wannier_centers, 'o')
ax.set_ylim([-1, 1])
ax.set_xlim([min(k2), max(k2)])
ax.set_xlabel(r'$k_2 / \pi$')
ax.set_ylabel(r'$\mathcal{v} / \pi$')
plt.show()