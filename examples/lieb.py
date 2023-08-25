import numpy as np

from band_topology.meshes.kspace import KSpace
from band_topology.models.tightbinding import TightBinding
from band_topology.topology import Topology
    
def lieb_kin(kspace_class, t=1, delta=0):
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

# Define the square Bravais lattice
lattice_vectors = [[1,0],[0,1]]
frac_high_symmetry_points = {'$\Gamma$':[0,0], '$X$':[0,1.0/2.0], '$M$':[1.0/2.0,1.0/2.0], '$\Gamma$@':[0,0]}

# The parameters for the lieb lattice
tb_parameters = dict(delta=0.3) # 0.3

# Define the meshgrid and tight-binding model
kspace = KSpace(lattice_vectors=lattice_vectors)
kspace.monkhorst_pack(nk_list=60)
tb = TightBinding(Hks_fnc=lieb_kin, kspace_class=kspace, tb_parameters=tb_parameters)
tb.plot_surface()

# Redefine on a path to get the spaghetti plots
path_kspace = KSpace(lattice_vectors=lattice_vectors)
path_kspace.path(special_points=frac_high_symmetry_points)
path_tb = TightBinding(Hks_fnc=lieb_kin, kspace_class=path_kspace, tb_parameters=tb_parameters)
path_tb.plot_path()

# Calculate the topology properties in flat band in the middle
top = Topology(tb=tb, subspace=[1])
print(f'Chern number = {top.chern_number()}')
print(f'metric number = {top.metric_number()}')

# Plot the xx elements of the geometric tensor
#top.plot_contour(function=top.quantum_metric, i=0, j=1, label='$g$') # to only plot a single element of the tensor
top.plot_contour(function=top.quantum_metric, label='$g$')
top.plot_colormesh(function=top.quantum_metric, label='$g$')
top.plot_contour(function=top.berry_curvature, label='$\Omega$')
top.plot_colormesh(function=top.berry_curvature, label='$\Omega$')