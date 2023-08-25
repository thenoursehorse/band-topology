import numpy as np

from band_topology.meshes.kspace import KSpace
from band_topology.models.tightbinding import TightBinding
from band_topology.topology import Topology
    
def honeycomb_kin(kspace_class, t=1, Delta=0, t2=0): #Delta=0.2, t2=0, 0.12, 0.3
    a1 = kspace_class.lattice_vectors[0]
    a2 = kspace_class.lattice_vectors[1]
    a3 = -(a1+a2)
    
    n1 = (a2 - a1) / 3.0
    n2 = (a3 - a2) / 3.0
    n3 = (a1 - a3) / 3.0
    ns = [n1, n2, n3]
    gammaks = 0
    for n in ns:
        gammaks += np.exp( 1j * kspace_class.cart_list @ n )

    # FIXME these must be wrong because the parameters are wrong for the critical point when chern != 0
    m1 = n2 - n3
    m2 = n3 - n2
    m3 = n1 - n3
    ms = [m1, m2, m3]
    alphaks = 0
    for m in ms:
        alphaks += np.sin(kspace_class.cart_list @ m )

    orb_dim = 2
    Hks = np.zeros(shape=(kspace_class.nks, orb_dim, orb_dim), dtype=np.complex_)
    Hks[...,0,0] = Delta + 2*t2*alphaks
    Hks[...,1,1] = -Delta - 2*t2*alphaks
    Hks[...,0,1] = t * gammaks
    Hks[...,1,0] = t * gammaks.conj()
    return kspace_class.to_mesh(Hks, A_type='array')

# Define the hexagonal Bravais lattice
lattice_vectors = [[-1.0/2.0, np.sqrt(3)/2.0], [-1.0/2.0, -np.sqrt(3)/2.0]]

# The parameters for the Haldane model to be a Chern insulator
tb_parameters = dict(Delta=0.2, t2=0.3)

# Define the meshgrid and tight-binding model for a visualization of the band structure
kspace = KSpace(lattice_vectors=lattice_vectors)
kspace.monkhorst_pack(nk_list=60, basis='cartesian', domain=[-2*np.pi,2*np.pi])
tb = TightBinding(Hks_fnc=honeycomb_kin, kspace_class=kspace, tb_parameters=tb_parameters)
tb.plot_contour(band=0)
tb.plot_surface()

# Redefine on a path to get the spaghetti plots
# The path uses a dictionary that has to have unique keys, so we use the @ char to distinguish them, and all
# @ get removed for printing/plotting
frac_high_symmetry_points = {'$\Gamma$':[0,0], '$M$':[1.0/2.0,0], '$K$':[1.0/3.0,1.0/3.0], '$\Gamma$@':[0,0]}
path_kspace = KSpace(lattice_vectors=lattice_vectors)
high_symmetry_points = dict()
for key,val in frac_high_symmetry_points.items():
    high_symmetry_points[key] = path_kspace.transform_ks(val, to_basis='cartesian')
path_kspace.path(special_points=high_symmetry_points, basis='cartesian')
#path_kspace.path(special_points=frac_high_symmetry_points, basis='fractional') # Can use either basis, same answer
path_tb = TightBinding(Hks_fnc=honeycomb_kin, kspace_class=path_kspace, tb_parameters=tb_parameters)
path_tb.plot_path()

# Redefine on a mesh in fractional coordinates corresponding to the a single BZ
frac_kspace = KSpace(lattice_vectors=lattice_vectors)
frac_kspace.monkhorst_pack(nk_list=60)
frac_tb = TightBinding(Hks_fnc=honeycomb_kin, kspace_class=frac_kspace, tb_parameters=tb_parameters)
frac_tb.plot_contour()
frac_tb.plot_surface()

# Calculate the topology properties in the bottom band
top = Topology(tb=frac_tb, subspace=[0], delta=1e-4)
print(f'Chern number = {top.chern_number()}')
print(f'metric number = {top.metric_number()}')

# Plot the xy elements of the geometric tensor
top.plot_contour(function=top.quantum_metric, label='$g$')
top.plot_contour(function=top.berry_curvature, label='$\Omega$')
top.plot_colormesh(function=top.quantum_metric, label='$g$')
top.plot_colormesh(function=top.berry_curvature, label='$\Omega$')