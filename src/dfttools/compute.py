# Uses boltztrap to interpolates using raw DFT data and then generates files for DOS, TDF and mu(T)

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

# -------------------------------------------------------------


def load_interpolation(dft_data_dir, bt2filnam, niter):
    """
    
    """
    if os.path.isfile(bt2filnam):
        print("Loading the precalculated results from", bt2filnam)
        data, equivalences, coeffs, metadata = serialization.load_calculation(
            bt2filnam
        )
        print("done")



    else:
        print("No pregenerated bt2 file found; performing a new interpolation")
        data = BTP.DFTData(dft_data_dir)
        equivalences = sphere.get_equivalences(
            data.atoms, data.magmom, niter * len(data.kpoints)
        )
        print(
            "There are",
            len(equivalences),
            "equivalence classes in the output grid",
        )
        coeffs = fite.fitde3D(data, equivalences)
        serialization.save_calculation(
            bt2filnam,
            data,
            equivalences,
            coeffs,
            serialization.gen_bt2_metadata(data, data.mommat is not None),
        )
    return data, equivalences, coeffs


def mu_of_T_compute(dft_data_dir, bt2filnam, niter):

    data, equivalences, coeffs = load_interpolation(dft_data_dir, bt2filnam, niter)
    
    lattvec = data.get_lattvec()
    eband, vvband, cband = fite.getBTPbands(
        equivalences, coeffs, lattvec, curvature=False
    )


    Cepsilon, Cdos, Cvvdos, cdos = BL.BTPDOS(eband, vvband, npts=n_bins)


    # 0.02 Ha = 0.54 eV
    mur = np.linspace(-.02, .02, 5000) + Efermi_DFT


    N, L0, L1, L2, Lm11 = BL.fermiintegrals(
        Cepsilon, Cdos, Cvvdos, mur=mur, Tr=Temperatures
    )




    print("computing for Nstar=", Nstar)
    fig, axes1 = plt.subplots(1, 2, dpi=150, figsize=(10,4))

    ax = axes1[0]
    # plt.plot(mur, N.T)
    ax.plot(mur/units.eV, N[0], 'o', label="T={}".format(Temperatures[0]))
    ax.plot(mur/units.eV, N[-1], 's', label="T={}".format(Temperatures[-1]))
    ax.axvline(Efermi_DFT/units.eV, 0, 1, color='green', label=r"$E_F\, DFT$")
    ax.axhline(-206, 0, 1, color='green')

    mu_of_T = []
    for i, Nstar in enumerate(NstarList):
        ifermi_list = [np.nonzero(N[i]<= Nstar)[0][0] for i in range(len(Temperatures))]
        Nvalues = [N[k, i] for k in range(len(Temperatures)) for i in ifermi_list]
        mu_values = [mur[i] for i in ifermi_list]
        mu_values = np.array(mu_values)
        mu_of_T.append(mu_values)
        # print(mu_values)

        if i == 0:
            ax.axhline(Nvalues[0], 0, 1, color='black')
            ax.axhline(Nvalues[-1], 0, 1, color='black')
            ax.axvline(mu_values[0]/units.eV, 0, 1, color='red')
            ax.axvline(mu_values[-1]/units.eV, 0, 1, color='red')

            ax.set_xlim(np.array([-0.7, 0.7]) + Efermi_DFT/units.eV)
            ax.set_ylim(np.array([-1, 1]) + NstarDFT)
            ax.legend()
            # plt.xlabel(r"$\mu=\epsilon-\epsilon_{DFT} \, [Ha]$")
            ax.set_xlabel(r"$\epsilon \, [eV]$")
            ax.set_ylabel(r"$Electron\, Count,\ N$")


        # ifermi = np.nonzero(mur >= Efermi_DFT)[0][0]
        # print(mur[ifermi])


            
            x, y = Temperatures, (mu_values-Efermi_DFT)/units.eV * 1000
            print("shift needed to EF (eV) = ", np.min(y))
            # y[:2] = np.min(y)
            coeff = np.polyfit(x, y, 3)
            polynomial = np.poly1d(coeff)
            # print(coeff)
            ax = axes1[1]
            ax.plot(x, y, label="Our DFT")
            ax.plot(x, polynomial(x), label="Our DFT fit")
            ax.set_ylabel(r"$\mu - E_F \, [m eV]$")
            ax.set_xlabel(r"T(K)")
            pass
        pass
    mu_of_T = np.array(mu_of_T)
    Temperatures = np.array(Temperatures, shape=(-1,1))
    print(mu_of_T.shape)
    
    combined = np.hstack((Temperatures, mu_of_T))


    # Write to file
    np.savetxt(dft_data_dir + signature + "_mu_of_T.csv", combined, fmt="%.6f", delimiter=",")


    pass


def dos_TDF_compute(dft_data_dir, bt2filnam, niter):

    data, equivalences, coeffs = load_interpolation(dft_data_dir, bt2filnam, niter)

    lattvec = data.get_lattvec()
    eband, vvband, cband = fite.getBTPbands(
        equivalences, coeffs, lattvec, curvature=False
    )



    
    erange = np.array([-5,5])*units.eV + Efermi_DFT
    Cepsilon, Cdos, Cvvdos, cdos = BL.BTPDOS(eband, vvband, npts=n_bins, erange=erange)


    const = 1.0
    Mvvdos = const * Cvvdos/Cdos
    Mvvdos[np.isnan(Mvvdos)] = 0.0


    # Unit conversion to eV
    energy = (Cepsilon - Efermi_DFT)/units.eV
    DOS = Cdos * units.eV
    TDF_CRTA = Cvvdos
    TDF_IDOS = Cvvdos / DOS


    np.savetxt(dft_data_dir + signature + "_DOS.csv", np.c_[energy, DOS], 
            delimiter=",",
            header="#(E-E_F) [eV], DOS [eV^-1], {}bins in range [-5,5] eV, m={} interpolation. {}".format(n_bins, niter, signature))


    np.savetxt(dft_data_dir + signature + "_TDF_CRTA.csv", np.c_[energy,TDF_CRTA[0,0,:],TDF_CRTA[1,1,:],TDF_CRTA[2,2,:]], 
            delimiter=",",
            header="#(E-E_F) [eV], TDF_xx, TDF_yy, TDF_zz, for CRTA, {}bins in range [-5,5] eV, m={} interpolation. {}".format(n_bins, niter, signature))

    np.savetxt(dft_data_dir + signature + "_TDF_IDOS.csv", np.c_[energy,TDF_IDOS[0,0,:],TDF_IDOS[1,1,:],TDF_IDOS[2,2,:]], 
            delimiter=",",
            header="#(E-E_F) [eV], TDF_xx, TDF_yy, TDF_zz, for Inverse DOS, {}bins in range [-5,5] eV, m={} interpolation. {}".format(n_bins, niter, signature))





    # Plot settings
    xmin, xmax = -0.4, 0.4

    fig, axes = plt.subplots(
        4, 1,
        figsize=(8, 10),
        sharex=True,
        gridspec_kw={"hspace": 0.05}
    )

    # DOS
    axes[0].plot(energy, DOS, lw=1.5)
    axes[0].set_ylabel("DOS")
    axes[0].axvline(0, linestyle="--", linewidth=1)

    # TDF_xx
    axes[1].plot(energy, TDF_CRTA[0, 0,:], lw=1.5, label="CRTA")
    axes[1].plot(energy, TDF_IDOS[0, 0,:], lw=1.5, label="IDOS")
    axes[1].set_ylabel(r"TDF$_{xx}$")
    axes[1].axvline(0, linestyle="--", linewidth=1)

    # TDF_yy
    axes[1].plot(energy, TDF_CRTA[1, 1, :], lw=1.5, label="CRTA")
    axes[1].plot(energy, TDF_IDOS[1, 1, :], lw=1.5, label="IDOS")
    axes[2].set_ylabel(r"TDF$_{yy}$")
    axes[2].axvline(0, linestyle="--", linewidth=1)

    # TDF_zz
    axes[1].plot(energy, TDF_CRTA[2, 2, :], lw=1.5, label="CRTA")
    axes[1].plot(energy, TDF_IDOS[2, 2, :], lw=1.5, label="IDOS")
    axes[3].set_ylabel(r"TDF$_{zz}$")
    axes[3].set_xlabel("Energy (eV)")
    axes[3].axvline(0, linestyle="--", linewidth=1)

    # Common x-range
    axes[3].set_xlim(xmin, xmax)

    plt.tight_layout()
    plt.savefig(fig_out_dir + "dos_TDF.png")




if __name__ == "__main__":

    dos_TDF_compute(dft_data_dir, bt2filnam, niter)
    mu_of_T_compute(dft_data_dir, bt2filnam, niter)


    pass
