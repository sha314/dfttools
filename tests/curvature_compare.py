# %% [markdown]
# # Compare inverse of effective mass computed by 
# (A) modified Fortran routine in QE after SCF or NSCF 
# (B) BoltzTraP2 interpolation + code
# 
# 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings




import numpy as np
from pathlib import Path
import argparse

import numpy as np
import plotly.graph_objects as go
from skimage import measure
from BoltzTraP2 import units

from dfttools import fermisurfer
from dfttools.units import *
from dfttools import fermi_boltztrap2
from dfttools import compute


# Read Fermisurfer file produced by (A) where energy is in Ry unit and curvature is in Ry-Bohr^2 unit
# 
# Write new Fermisurfer file where energy is in eV and curvature is in Ha-Bohr^2 unit so that unit of 1/m* is m_e/m*


DATA_PATH = "./FermiSurferInputs/qe/nscf_fermi/"
filename = DATA_PATH + "nb3s4_inv_mstar_xx.frmsf"
result = fermisurfer.read_frmsf (filename)
filename = DATA_PATH + "eV-Ha/nb3s4_inv_mstar_xx-eV.frmsf"
fermisurfer.write_frmsf(filename, result, transformEnergy=lambda x: x*Ry_to_eV, transformColor=lambda x: x*Ry_to_Ha)


# Now read the new fermisurfer file
filename = DATA_PATH + "eV-Ha/nb3s4_inv_mstar_xx-eV.frmsf"
result = fermisurfer.read_frmsf (filename)


# need to convert from (nband, nk1, nk2, nk3) shape to (nband, nk1*nk2*nk3) and then transpose 
mat = result['scalar']
eig = result['energy']
kpts = []
nk1, nk2, nk3 = result['nk']
scalar = []
energy = []
for i1 in range(nk1):
    for i2 in range(nk2):
        for i3 in range(nk3):
            kpts.append([i1/nk1, i2/nk2, i3/nk3])
            scalar.append(mat[:, i1, i2, i3])
            energy.append(eig[:, i1, i2, i3])
        pass
    pass
surfer_energy = np.array(energy).T
surfer_scalar = np.array(scalar).T
surfer_kpts = np.array(kpts)

# # Now do (B) Boltztrap interpolation and computation of curvature


npoints = 20 # same number of k points for the plot


# You might wanna change the directory/filename of the Boltztrap interpolation
dft_data_dir = "./PBEsol-Relaxed/"
niter = 20
bt2filnam = dft_data_dir + "Nb3S4_BLZTRP_m{}.bt2".format(niter)

data, equivalences, coeffs = compute.load_interpolation(dft_data_dir, bt2filnam, niter)

# Fermi energy and cell vector
efermi = data.fermi
efermi_ev = efermi/units.eV
cell = data.atoms.cell

# x data for plot
x = np.linspace(0, 1, npoints)



# Kpath 1
kpath = "[0.0,0.0,0.0], [.5, 0.0, 0.0]"

fig, axes = plt.subplots(2, 1, figsize=(6, 10), dpi=100)

k, e = fermi_boltztrap2.extract_kpath_data(surfer_kpts, surfer_energy, cell, kpath, [npoints], 0)
axes[0].plot(x, e[0].T, label="Fermisurfer", linestyle="--")

kpoints, energy, velocity, curvature = fermi_boltztrap2.extract_kpath_interpolate(
    data, equivalences, coeffs, kpath, [npoints])
axes[0].plot(x, (energy[0][(61,62,63,64),:].T-efermi)/units.eV, label="BoltzTraP2", linestyle="-")

axes[0].set_ylabel(r"$E-E_F [Ha]$")
axes[0].legend()



k, e = fermi_boltztrap2.extract_kpath_data(surfer_kpts, surfer_scalar, cell, kpath, [npoints], 0)
axes[1].plot(x, e[0].T, label="Fermisurfer", linestyle="--")
axes[1].plot(x, curvature[0][0,0,(61,62,63,64),:].T, label="BoltzTraP2", linestyle="-")
axes[1].set_ylabel(r"$m_e/m_* [Ha\cdot Bohr^2]$")
axes[1].legend()
# plt.ylim(-0.5, 0.5)
plt.show()

# Kpath 2
kpath = "[0.0,0.0,0.0], [.0, 0.0, 0.5]"

fig, axes = plt.subplots(2, 1, figsize=(6, 10), dpi=100)

k, e = fermi_boltztrap2.extract_kpath_data(surfer_kpts, surfer_energy, cell, kpath, [npoints], 0)
axes[0].plot(x, e[0].T, label="Fermisurfer", linestyle="--")

kpoints, energy, velocity, curvature = fermi_boltztrap2.extract_kpath_interpolate(
    data, equivalences, coeffs, kpath, [npoints])
axes[0].plot(x, (energy[0][(61,62,63,64),:].T-efermi)/units.eV, label="BoltzTraP2", linestyle="-")

axes[0].set_ylabel(r"$E-E_F [Ha]$")
axes[0].legend()



k, e = fermi_boltztrap2.extract_kpath_data(surfer_kpts, surfer_scalar, cell, kpath, [npoints], 0)
axes[1].plot(x, e[0].T, label="Fermisurfer", linestyle="--")
axes[1].plot(x, curvature[0][0,0,(61,62,63,64),:].T, label="BoltzTraP2", linestyle="-")
axes[1].set_ylabel(r"$m_e/m_* [Ha\cdot Bohr^2]$")
axes[1].legend()
# plt.ylim(-0.5, 0.5)
plt.show()

# Kpath 3
kpath = "[0.0,0.0,0.0], [.5, 0.5, 0.5]"

fig, axes = plt.subplots(2, 1, figsize=(6, 10), dpi=100)

k, e = fermi_boltztrap2.extract_kpath_data(surfer_kpts, surfer_energy, cell, kpath, [npoints], 0)
axes[0].plot(x, e[0].T, label="Fermisurfer", linestyle="--")

kpoints, energy, velocity, curvature = fermi_boltztrap2.extract_kpath_interpolate(
    data, equivalences, coeffs, kpath, [npoints])
axes[0].plot(x, (energy[0][(61,62,63,64),:].T-efermi)/units.eV, label="BoltzTraP2", linestyle="-")

axes[0].set_ylabel(r"$E-E_F [Ha]$")
axes[0].legend()



k, e = fermi_boltztrap2.extract_kpath_data(surfer_kpts, surfer_scalar, cell, kpath, [npoints], 0)
axes[1].plot(x, e[0].T, label="Fermisurfer", linestyle="--")
axes[1].plot(x, curvature[0][0,0,(61,62,63,64),:].T, label="BoltzTraP2", linestyle="-")
axes[1].set_ylabel(r"$m_e/m_* [Ha\cdot Bohr^2]$")
axes[1].legend()
# plt.ylim(-0.5, 0.5)
plt.show()




