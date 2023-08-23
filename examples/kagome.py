import numpy as np

from band_topology.meshes.kspace import KSpace
from band_topology.models.tightbinding import TightBinding
from band_topology.topology import Topology
    
def kagome_kin(kspace_class, t=1):
    a1 = kspace_class.lattice_vectors[0]
    a2 = kspace_class.lattice_vectors[1]
    a3 = -(a1+a2)

    orb_dim = 3
    Hks = np.zeros(shape=(kspace_class.nks, orb_dim, orb_dim), dtype=np.complex_)
    Hks[...,0,1] = 2.0 * t * np.cos( 0.5 * kspace_class.cart_list @ a2 )
    Hks[...,0,2] = 2.0 * t * np.cos( 0.5 * kspace_class.cart_list @ a1 )
    Hks[...,1,2] = 2.0 * t * np.cos( 0.5 * kspace_class.cart_list @ a3 )
    Hks[...,1,0] = Hks[...,0,1].conj()
    Hks[...,2,0] = Hks[...,0,2].conj()
    Hks[...,2,1] = Hks[...,1,2].conj()
    return kspace_class.to_mesh(Hks, A_type='array')

# Define the hexagonal Bravais lattice
lattice_vectors = [[-1.0/2.0, np.sqrt(3)/2.0], [-1.0/2.0, -np.sqrt(3)/2.0]]
high_symmetry_points = {'G':[0,0], 'M':[1.0/2.0,0], 'K':[1.0/3.0,1.0/3.0], 'G2':[0,0]}

# Define the meshgrid and tight-binding model for a visualization of the band structure
kspace = KSpace(lattice_vectors=lattice_vectors)
kspace.monkhorst_pack(nk_list=60, basis='cartesian', domain=[-2*np.pi,2*np.pi])
tb = TightBinding(Hks_fnc=kagome_kin, kspace_class=kspace)
tb.plot_contour(band=1)
tb.plot_surface()

# Redefine on a mesh in fractional coordinates corresponding to the a single BZ
frac_kspace = KSpace(lattice_vectors=lattice_vectors)
frac_kspace.monkhorst_pack(nk_list=60)
frac_tb = TightBinding(Hks_fnc=kagome_kin, kspace_class=frac_kspace)
frac_tb.plot_contour(band=1)
frac_tb.plot_surface()

# Calculate the topology properties in the bottom band
top = Topology(tb=frac_tb, subspace=[1])
print(f'Chern number = {top.chern_number()}')
print(f'metric number = {top.metric_number()}')

# Plot the xy elements of the geometric tensor
top.plot_contour(i=0, j=1)
top.plot_colormesh(i=0, j=1)