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

def h_k(kspace_class, t=1, t_sp1=0, t_sp2=0, eps=0):
    t_s = t/2.0
    t_p = -t/2.0

    def kay(i):
        return kspace_class.klist('crystal')[:,i]
    
    Hks = eps * pauli('z') * np.ones(shape=kay(0).shape)[...,None,None] \
        + t_sp1 * pauli('y') * np.sin(kay(0))[...,None,None] \
        + (t_s - t_p) * pauli('z') * np.cos(kay(0))[...,None,None] \
        + (t_s + t_p) * pauli('id') * np.cos(kay(0))[...,None,None] \
        + t_sp2 * pauli('x') * np.sin(kay(0))[...,None,None]
    return kspace_class.to_mesh(Hks, A_type='array')

# Define the square Bravais lattice
lattice_vectors = [[1]]
frac_high_symmetry_points = {'$\Gamma$':[0], '$X$':[1.0/2.0]}

# The parameters of the model
#tb_parameters = dict(t=0, t_sp1=0, eps=1) # trivial
tb_parameters = dict(t=1, t_sp1=1, eps=0) # topological

# Define the meshgrid and tight-binding model
kspace = KSpace(lattice_vectors=lattice_vectors)
kspace.monkhorst_pack(nk_list=100)
tb = TightBinding(Hks_fnc=h_k, kspace_class=kspace, tb_parameters=tb_parameters)

# Redefine on a path to get the spaghetti plots
path_kspace = KSpace(lattice_vectors=lattice_vectors)
path_kspace.path(special_points=frac_high_symmetry_points)
path_tb = TightBinding(Hks_fnc=h_k, kspace_class=path_kspace, tb_parameters=tb_parameters)
path_tb.plot_path()

# Calculate the topology properties in flat band in the middle
subspace=[0]
top = Topology(tb=tb, subspace=subspace)

# Calculate the Wilson matrix to get the winding numbers
W = top.wilson_path(n_points=100, path={'0':[0], 'pi':[1]})

e, v = np.linalg.eig(W)
wannier_centers = np.sort(np.angle(e), axis=-1) / (2*np.pi)
print("Berry phase =", wannier_centers * 2 * np.pi)