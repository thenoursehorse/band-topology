import numpy as np

from band_topology.meshes.kspace import KSpace
from band_topology.models.tightbinding import TightBinding
from band_topology.topology import Topology
    
def lieb_kin(kspace_class, t=1, delta=0.3):
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

# Define the meshgrid and tight-binding model
kspace = KSpace(lattice_vectors=lattice_vectors)
kspace.monkhorst_pack(nk_list=60)
tb = TightBinding(Hks_fnc=lieb_kin, kspace_class=kspace)
tb.plot_surface()

# Calculate the topology properties in flat band in the middle
top = Topology(tb=tb, subspace=[1])
print(f'Chern number = {top.chern_number()}')
print(f'metric number = {top.metric_number()}')

# Plot the xx elements of the geometric tensor
top.plot_contour(i=0, j=0)
top.plot_colormesh(i=0, j=0)