import numpy as np

from band_topology.meshes.kspace import KSpace
from band_topology.models.tightbinding import TightBinding
from band_topology.topology import Topology
    
def hypercubic_kin(kspace_class, t=1, orb_dim=1):
    ks = kspace_class.mesh('cartesian')
    di = np.diag_indices(orb_dim)
    Hks = np.zeros(shape=(*ks[0].shape, orb_dim, orb_dim))
    Hks[...,di[0],di[1]] = 2.0 * t * np.sum(np.cos(ks), axis=0)[..., None]
    return Hks

chain_lattice_vectors = [[1]]
chain_kspace = KSpace(lattice_vectors=chain_lattice_vectors)
chain_kspace.monkhorst_pack()
chain = TightBinding(Hks_fnc=hypercubic_kin, kspace_class=chain_kspace)

# Quick test on the square lattice
square_lattice_vectors = [[1,0],[0,1]]
square_high_symmetry_points = {'G':[0,0], 'X':[1,0], 'M':[1,1], 'G2':[0,0]}
square_kspace = KSpace(lattice_vectors=square_lattice_vectors)
square_kspace.monkhorst_pack(nk_list=60)
square = TightBinding(Hks_fnc=hypercubic_kin, kspace_class=square_kspace)
square.plot_contour()
square.plot_surface()

cubic_lattice_vectors = [[1,0,0],[0,1,0],[0,0,1]]
cubic_kspace = KSpace(lattice_vectors=cubic_lattice_vectors)
cubic_kspace.monkhorst_pack()
cubic = TightBinding(Hks_fnc=hypercubic_kin, kspace_class=cubic_kspace)