# Uses boltztrap to interpolates using raw DFT data and then generates DOS.

import copy
import os.path

import ase
import ase.io
import matplotlib.pylab as plt
import numpy as np



import BoltzTraP2.bandlib as BL
import BoltzTraP2.dft as BTP
import BoltzTraP2.io as IO
from BoltzTraP2 import fite, serialization, sphere, units
from BoltzTraP2.misc import ffloat


import contextlib
import copy
import os.path
import sys
from pathlib import Path



# -------------------------------------------------------------

dft_data_dir = "./"
Efermi_DFT = 0
niter = 20
bt2filnam = dft_data_dir + "Nb3S4_BLZTRP_m{}.bt2".format(niter)
n_bins = 20_000
signature = "Nb3S4_PBEsol_NCPP_QE_data"
Temperatures = np.linspace(10.0, 500.0, num=50)

NstarDFT = -126
NstarList = [-126.1, -126.15]






fig_out_dir = "May10m_{}_bins_{}".format(niter, n_bins)
Path(fig_out_dir).mkdir(exist_ok=True)

bt2filnam = dft_data_dir + "Nb3S4_BLZTRP_m{}.bt2".format(niter)

filename_DOS = dft_data_dir      + "Nb3S4_PBEsol_NCPP_QE_data24_DOS_m_{}_bins_{}.csv".format(niter, n_bins)
filename_TDF_CRTA = dft_data_dir +  "Nb3S4_PBEsol_NCPP_QE_data24_TDF_CRTA_m_{}_bins_{}.csv".format(niter, n_bins)
filename_TDF_IDOS = dft_data_dir +  "Nb3S4_PBEsol_NCPP_QE_data24_TDF_IDOS_m_{}_bins_{}.csv".format(niter, n_bins)


# -------------------------------------------------------------


from BoltzTraP2.units import *

BOLTZMANN_EV = BOLTZMANN/units.eV
print(BOLTZMANN_EV)

def FD(E, mu, T):
    kT =  T*BOLTZMANN_EV
    return 1.0/(np.exp((E-mu)/kT) + 1.0)

def dFD_dE(E, mu, T):
    kT =  T*BOLTZMANN_EV
    f = FD(E, mu, T)
    return -f*(1-f)/kT

def fermi_integral(E_values, TDF_values, T, mu):
    dE = E_values[1]-E_values[0]
    int0 = -TDF_values*dFD_dE(E_values, mu, T)
    L0 = np.sum(int0)*dE
    int1 = int0*(E_values-mu)
    L1 = np.sum(int1)*dE
    return L0, L1


def shift_and_clip(E_values, deltaE, erange=(-0.4,0.4)):
    x = E_values - deltaE
    idx = np.logical_and(x <= erange[1], x >= erange[0])
    x = x[idx]
    return x, idx

def get_OnsagerCoeff(E_values, TDF_values, Tlist, mulist=None):
    if mulist is None:
        mulist = np.copy(Tlist)*0.0
        pass

    tmp = []
    for i, T in enumerate(Tlist):
        L0, L1 = fermi_integral(E_values, TDF_values, T, mulist[i])
        tmp.append([L0, L1])

    return np.array(tmp)

def get_thermopower(E_values, TDF_values, Tlist, mulist=None):
    if mulist is None:
        mulist = np.copy(Tlist)*0.0
        pass

    tmp = []
    for i, T in enumerate(Tlist):
        L0, L1 = fermi_integral(E_values, TDF_values, T, mulist[i])
        tmp.append(L1/L0/(-T))

    return np.array(tmp)


def get_tau_from_dos(energy_, dos_, erange=(-5,5)):
    """
    Get scattering time tau from dos in a given range.

    If there are zero's in the DOS, using L2 regularization we avoid that. Effect is that there could be unwanted peaks on
    either side of the zero but that shouldn't change overall behavior of computed transport coeff.
    
    """
    SCALE = 1e-2
    if erange is not None:
        idx = np.logical_and(energy_ >= erange[0], energy_ <= erange[1])
        x = energy_[idx]
        y = dos_[idx]
    else:
        x=energy_
        y = dos_
    
    if abs(y.min() - 0) < 1e-3:
        lam = SCALE * y.max()
        tau = y/(y**2 + lam)
        pass
    else:
        tau = 1/y
        pass

    tau[np.isnan(tau)] = 0.0

    return x, tau




def compute_Szz_Sxx():

    energy, DOS = np.loadtxt(filename_DOS, delimiter=',').T

    energy1, TDF_xx_CRTA, TDF_yy_CRTA, TDF_zz_CRTA = np.loadtxt(filename_TDF_CRTA, delimiter=',').T

    energy2, TDF_xx_IDOS, TDF_yy_IDOS, TDF_zz_IDOS = np.loadtxt(filename_TDF_IDOS, delimiter=',').T

    Tzzdata, Szzdata = np.loadtxt("../thermopower.txt").T
    T1xxdata, S1xxdata = np.loadtxt("../thermopower.txt").T


    for shift in range(-50, 60, 5):
        Ep=shift/1000
        x, idx = shift_and_clip(energy, Ep, erange=(-0.6, 0.6))
        TDF_values = TDF_zz_IDOS[idx]
        DOS_values = DOS[idx]

        fig, axes = plt.subplots(2, 2, figsize=(24, 5), dpi=200)
        axes = axes.flatten()


        ax = axes[0]
        ax.plot(energy, DOS, label="DOS")
        ax.plot(x, DOS_values, label="DOS, shifted")
        ax.set_xlabel("(E-E_F-{}meV ) (eV)".format(Ep*1000))
        ax.axvline(0, 0, 10, color='red')
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim(0, 8)
        ax.legend()


        ax = axes[1]
        ax.plot(energy, TDF_zz_IDOS, label="TDF")
        ax.plot(x, TDF_values, label="TDF, shifted")
        ax.plot(energy, -dFD_dE(energy, 0, 300)/30, 'k--', label="-df/dE(T=300K)/30")
        ax.axvline(0, 0, 10, color='red')
        ax.set_xlabel("(E-E_F-{}meV ) (eV)".format(Ep*1000))
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim(0, 0.7)
        ax.legend()


        ax = axes[2]
        ax.plot(Tzzdata, Szzdata, 'o', label="Experiment", alpha=0.6)



        
        Tlist = np.linspace(10, 400, 20)
        SofT = get_thermopower(x, TDF_values, Tlist)
        ax.plot(Tlist, SofT*10**6, label=r"$\mu(T)=0$, Org.")
        ax.set_ylim(-60, 20)
        ax.set_xlabel(r"Temperature (K)")
        ax.set_ylabel(r"Thermopwoer, $S_zz\, \mu V/K$")
        ax.legend()



        ax = axes[3]
        ax.plot(T1xxdata, S1xxdata, 'o', label="Experiment", alpha=0.6)



        
        Tlist = np.linspace(10, 400, 20)
        SofT = get_thermopower(x, TDF_values, Tlist)
        ax.plot(Tlist, SofT*10**6, label=r"$\mu(T)=0$, Org.")
        ax.set_ylim(-60, 20)
        ax.set_xlabel(r"Temperature (K)")
        ax.set_ylabel(r"Thermopwoer, $S_zz\, \mu V/K$")
        ax.legend()

        plt.savefig(fig_out_dir + "/Nb3S4_m_{}_bins_{}_Ep_{:.2f}meV.png".format(niter, n_bins, Ep*1000))
    # break



def compute_Szz():

    energy, DOS = np.loadtxt(filename_DOS, delimiter=',').T

    energy1, TDF_xx_CRTA, TDF_yy_CRTA, TDF_zz_CRTA = np.loadtxt(filename_TDF_CRTA, delimiter=',').T

    energy2, TDF_xx_IDOS, TDF_yy_IDOS, TDF_zz_IDOS = np.loadtxt(filename_TDF_IDOS, delimiter=',').T

    Tdata, Sdata = np.loadtxt("../thermopower.txt").T
    


    for shift in range(-10, 60, 1):
        Ep=shift/1000
        x, idx = shift_and_clip(energy, Ep, erange=(-0.6, 0.6))
        TDF_values = TDF_zz_IDOS[idx]
        DOS_values = DOS[idx]

        fig, axes = plt.subplots(1, 3, figsize=(24, 5), dpi=200)
        axes = axes.flatten()


        ax = axes[0]
        ax.plot(energy, DOS, label="DOS")
        ax.plot(x, DOS_values, label="DOS, shifted")
        ax.set_xlabel("(E-E_F-{}meV ) (eV)".format(Ep*1000))
        ax.axvline(0, 0, 10, color='red')
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim(0, 8)
        ax.legend()


        ax = axes[1]
        ax.plot(energy, TDF_zz_IDOS, label="TDF")
        ax.plot(x, TDF_values, label="TDF, shifted")
        ax.plot(energy, -dFD_dE(energy, 0, 300)/30, 'k--', label="-df/dE(T=300K)/30")
        ax.axvline(0, 0, 10, color='red')
        ax.set_xlabel("(E-E_F-{}meV ) (eV)".format(Ep*1000))
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim(0, 0.7)
        ax.legend()


        ax = axes[2]
        ax.plot(Tdata, Sdata, 'o', label="Experiment", alpha=0.6)



        
        Tlist = np.linspace(10, 400, 20)
        SofT = get_thermopower(x, TDF_values, Tlist)
        ax.plot(Tlist, SofT*10**6, label=r"$\mu(T)=0$, Org.")
        ax.set_ylim(-60, 20)
        ax.set_xlabel(r"Temperature (K)")
        ax.set_ylabel(r"Thermopwoer, $\mu V/K$")
        ax.legend()

        plt.savefig(fig_out_dir + "/Nb3S4_m_{}_bins_{}_Ep_{:.2f}meV.png".format(niter, n_bins, Ep*1000))
    # break



if __name__ == "__main__":
    compute_Szz()
    # plot()
    