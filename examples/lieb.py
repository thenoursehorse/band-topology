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

def gell_mann(i):
    if i == 0:
        return np.eye(3)
    elif i == 1:
        return np.array( [ [0, 1, 0], [1, 0, 0], [0, 0, 0] ] )
    elif i == 4:
        return np.array( [ [0, 0, 1], [0, 0, 0], [1, 0, 0] ] )
    elif i == 6:
        return np.array( [ [0, 0, 0], [0, 0, 1], [0, 1, 0] ] )

def lieb_kin(kspace_class, t=1, tt=0, ttt=0, eps_A=0, Lambda=0, t_perp=0, J1=0, J2=0, stack='single', delta=0):
    #a1 = kspace_class.lattice_vectors[0]
    #a2 = kspace_class.lattice_vectors[1]
    #k1 = kspace_class.cart_list @ a1
    #k2 = kspace_class.cart_list @ a2
    k1 = kspace_class.crystal_list[:,0]
    k2 = kspace_class.crystal_list[:,1]

    orb_dim = 3
    Hks = np.zeros(shape=(kspace_class.nks, orb_dim, orb_dim), dtype=np.complex_)
    
    # FIXME for single layer all parameters seem correct except ttt (t'')

    Hks[...,0,0] = eps_A
    Hks[...,1,1] = - 4 * ttt * np.cos( k1 ) # FIXME paper says this is 2 ttt
    Hks[...,2,2] = - 4 * ttt * np.cos( k2 ) # FIXME paper says this is 2 ttt but it seems wrong
    
    Hks[...,0,1] = - 2 * t * np.cos( 0.5 * k1 ) - 2 * t * 1j * delta * np.sin( 0.5 * k1)
    Hks[...,0,2] = - 2 * t * np.cos( 0.5 * k2 ) - 2 * t * 1j * delta * np.sin( 0.5 * k2)
    Hks[...,1,2] = - 4 * tt * np.cos( 0.5 * k1) * np.cos( 0.5 * k2) \
              - 1j * 4 * Lambda * np.sin( 0.5 * k1 ) * np.sin( 0.5 * k2 )

    Hks[...,1,0] = Hks[...,0,1].conj()
    Hks[...,2,0] = Hks[...,0,2].conj()
    Hks[...,2,1] = Hks[...,1,2].conj()

    if stack == 'aa':
        Vks = - t_perp * gell_mann(0) * np.cos( 0.5 * k1 )[...,None,None] \
              - 2 * J1 * gell_mann(1) * np.cos( 0.5 * k1 )[...,None,None] \
              - 2 * J1 * gell_mann(4) * np.cos( 0.5 * k2 )[...,None,None] \
              - 4 * J2 * gell_mann(6) * ( np.cos( 0.5 * k1 ) * np.cos( 0.5 * k2 ) )[...,None,None]
        Hks = np.kron(Hks, pauli('id')) + np.kron(Vks, pauli('x'))

    if stack == 'ab':
        Vks = - 2 * J1 * gell_mann(0) * np.cos( 0.5 * k1 )[...,None,None] \
              - t_perp * gell_mann(1) \
              - 4 * J2 * gell_mann(4) * ( np.cos( 0.5 * k1 ) * np.cos( 0.5 * k2 ) )[...,None,None] \
              - 2 * J1 * gell_mann(6) * np.cos( 0.5 * k2 )[...,None,None]
        Hks = np.kron(Hks, pauli('id')) + np.kron(Vks, pauli('x'))

    if stack == 'ac':
        Vks = - 4 * J2 * gell_mann(0) * ( np.cos( 0.5 * k1 ) * np.cos( 0.5 * k2 ) )[...,None,None] \
              - 2 * J1 * gell_mann(1) * np.cos( 0.5 * k2 )[...,None,None] \
              - 2 * J1 * gell_mann(4) * np.cos( 0.5 * k1 )[...,None,None] \
              - t_perp * gell_mann(6)
        Hks = np.kron(Hks, pauli('id')) + np.kron(Vks, pauli('x'))
              
    return kspace_class.to_mesh(Hks, A_type='array')

# Define the square Bravais lattice
a1 = np.array([1,0])
a2 = np.array([0,1])
lattice_vectors = [a1, a2]
high_symmetry_points = {'$\Gamma$':[0,0], '$X$':[0,0.5], '$M$':[0.5,0.5], '$\Gamma$@':[0,0]}
#high_symmetry_points = {'$\Gamma$':[0,0], '$X_1$':[0,0.5], '$M$':[0.5,0.5], '$\Gamma$@':[0,0], '$X_2$':[0.5,0], '$M@':[0.5,0.5]}

# Position of lattice sites
qa = a1 - a1
qb = 0.5 * a1
qc = 0.5 * a2

# Pick the type of stacking
stack = 'ac'

if stack == 'single':
    shift = 0
elif stack == 'aa':
    shift = [0, 0]
elif stack == 'ab':
    shift = 0.5 * a1
elif stack == 'ac':
    shift = 0.5 * (a1 + a2)

# The parameters for the lieb lattice
#tb_parameters = dict(delta=0.3)
tb_parameters = dict(t=1, tt=0.3, ttt=0.2, eps_A=-1, Lambda=0.35, t_perp=0.45, J1=0.25, J2=0.15, stack=stack)

# Define the meshgrid and tight-binding model
kspace = KSpace(lattice_vectors=lattice_vectors)
kspace.monkhorst_pack(nk_list=60)
tb = TightBinding(Hks_fnc=lieb_kin, kspace_class=kspace, tb_parameters=tb_parameters)
tb.plot_surface()

# Redefine on a path to get the spaghetti plots
path_kspace = KSpace(lattice_vectors=lattice_vectors)
path_kspace.path(special_points=high_symmetry_points)
path_tb = TightBinding(Hks_fnc=lieb_kin, kspace_class=path_kspace, tb_parameters=tb_parameters)
path_tb.plot_path()

# Calculate the topology properties
#subspace=[0,1]
#subspace=[2,3]
subspace=[-2,-1]
top = Topology(tb=tb, subspace=subspace)
print(f'Chern number = {top.chern_number()}')
print(f'metric number = {top.metric_number()}')

# Plot the xx elements of the geometric tensor
#top.plot_contour(function=top.quantum_metric, i=0, j=1, label='$g$') # to only plot a single element of the tensor
top.plot_contour(function=top.quantum_metric, label='$g$')
top.plot_colormesh(function=top.quantum_metric, label='$g$')
top.plot_contour(function=top.berry_curvature, label='$\Omega$')
top.plot_colormesh(function=top.berry_curvature, label='$\Omega$')

# Work out V
q = [qa, qb, qc]
V = [np.zeros(shape=(tb.nbands, tb.nbands), dtype=np.complex_) for i in range(kspace.d)]
G = [top._kspace.reciprocal_vectors[0], top._kspace.reciprocal_vectors[1]]
for k in range(kspace.d):
    for i in range(len(q)):
        V[k][i,i] = np.exp(1j * G[k] @ q[i])
        if stack != 'single':
            q_shift = (q[i] + shift) % 1
            V[k][i+3,i+3] = np.exp(1j * G[k] @ q_shift)

# Calculate Wilson loops
#k2 = np.linspace(-0.5, 0.5, 100)
k2 = np.linspace(0, 1, 200)
W2 = np.empty(shape=(len(k2), len(subspace), len(subspace)), dtype=np.complex_)
for i,k in enumerate(k2):
    W2[i,...] = top.wilson_path(n_points=200, path={'0':[0,k], '2pi':[1,k]}, V=V[0])
e2, v2 = np.linalg.eig(W2)
wannier_centers_2 = np.sort(np.angle(e2), axis=-1) / (2*np.pi)

k1 = np.linspace(0, 1, 200)
W1 = np.empty(shape=(len(k1), len(subspace), len(subspace)), dtype=np.complex_)
for i,k in enumerate(k1):
    W1[i,...] = top.wilson_path(n_points=200, path={'0':[k,0], '2pi':[k,1]}, V=V[1])
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