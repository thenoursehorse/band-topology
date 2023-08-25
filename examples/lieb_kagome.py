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
a = 1
lattice_vectors = [[2.0*a, 0.0], [-1.0*a, np.sqrt(3)*a]]
frac_high_symmetry_points = {'$\Gamma$':[0,0], '$M$':[1.0/2.0,0], '$K$':[1.0/3.0,1.0/3.0], '$\Gamma$@':[0,0]}

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
top = Topology(tb=frac_tb, subspace=[3,4,5])
print(f'Chern number = {top.chern_number()}')
print(f'metric number = {top.metric_number()}')

# Plot the xx elements of the geometric tensor
top.plot_contour(function=top.quantum_metric, label='$g$')
top.plot_colormesh(function=top.quantum_metric, label='$g$')
top.plot_contour(function=top.berry_curvature, label='$\Omega$')
top.plot_colormesh(function=top.berry_curvature, label='$\Omega$')