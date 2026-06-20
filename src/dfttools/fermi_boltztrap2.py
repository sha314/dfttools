# Computes energy bands, group velocities and inverse of effective mass 
# using boltztrap interpolation along given symmetry path
# and plots it


from dfttools import compute
from dfttools.units import *
import numpy as np
import ase.dft.kpoints as asekp
import ast
import itertools


from BoltzTraP2.misc import TimerContext
from BoltzTraP2 import fite
import matplotlib.pyplot as plt




def generate_band_path(kpath, cell, nkpoints):
    """
    
    
    """
    band_path = asekp.bandpath(kpath, cell, nkpoints)
    if isinstance(band_path, asekp.BandPath):
        # For newer versions of ASE.
        kp = band_path.kpts
        # print("band_path.get_linear_kpoint_axis()")
        # print(band_path.get_linear_kpoint_axis())
        dkp, dcl = band_path.get_linear_kpoint_axis()[:2]
        # print("dkp, dcl")
        # print(dkp, dcl)
    else:
        # For older versions of ASE.
        kp, dkp, dcl = band_path
        pass
    return kp, dkp, dcl


def parse_one_k_path(kpath="[0.0,0.0,0.0], [0.5, 0.0, 0.0]"):
    """
    kpath : str
        k path as str in fractional coordiante.
        It can be two points coordinte: '[0.0,0.0,0.0], [0.5, 0.0, 0.0]'
        Or multiple points coordinates: '[0.0,0.0,0.0], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], [0.0,0.0,0.0], [0.0,0.0,0.5]'
    """
    try:
        kpaths = ast.literal_eval(kpath)
    except ValueError:
        print("cannot be parsed as a Python literal")

    kpaths = [
        list(group)
        for k, group in itertools.groupby(kpaths, key=lambda x: x is not None)
        if k
    ]

    # print(kpaths)
    kpaths = [np.array(i, dtype=np.float64) for i in kpaths]  
    return kpaths




def parse_k_path(kpath):
    # The second position alargument is first interpreted as a Python literal,
    # and after parsing it is cast to a NumPy array, which must have the right
    # dimensions. The special value None directs the parser to split the path
    # in several parts.

    kpath_list = []
    if isinstance(kpath, list) or isinstance(kpath, tuple):
        # each element of list is a k path, evaluate them independently
        kpath_list = [parse_one_k_path(kp) for kp in kpath]
    else:
        kpath_list = [parse_one_k_path(kpath)]
        pass
    return kpath_list



def extract_kpath_data(
    kpts_grid,           # (N_kpts, 3)  – your uniform grid k-points (fractional)
    energies_grid,       # (N_bands, N_kpts) – energies at those k-points
    cell,                # ASE cell or 3×3 array
    kpath,          # tuple(kpath_string, nkpoints_list) or None for default
    nkpoints_list,
    fermi=0.0,
    shift_eV=0.0,
    tol=1e-5,
):
    """
    Extract band-structure data along a k-path from a uniform energy grid.

    Returns
    -------
    segments_info : list[dict]
        [{"label": str, "first": int, "last": int}, ...]
        Indices refer to a flattened concatenation of all segments.
    kpaths_list : list[np.ndarray]
        [array(n_points, 3), ...] - 3D k-points along each segment
    energies_list : list[np.ndarray]
        [array(n_bands, n_points), ...] - energies for each segment
    """

    n_bands = energies_grid.shape[0]
    e_shift = fermi + shift_eV / Ha_to_eV
    energies = energies_grid - e_shift


    kpaths_raw = ast.literal_eval(kpath)
    print(type(kpaths_raw))
    if not (isinstance(kpaths_raw, list) or isinstance(kpaths_raw, tuple)):
        raise ValueError("kpath must parse to a Python list")

    # Split into segments (by None if present, else consecutive pairs)
    has_none = any(x is None for x in kpaths_raw)
    if has_none:
        kpath_segments = [
            list(g) for k, g in itertools.groupby(kpaths_raw, key=lambda x: x is not None) if k
        ]
    else:
        kpath_segments = [
            [kpaths_raw[i], kpaths_raw[i + 1]]
            for i in range(len(kpaths_raw) - 1)
        ]

    kpath_segments = [np.array(seg, dtype=np.float64) for seg in kpath_segments]
    n_seg = len(kpath_segments)

    # Ensure nkpoints_list matches number of segments
    if len(nkpoints_list) < n_seg:
        nkpoints_list = np.concatenate([
            nkpoints_list,
            np.full(n_seg - len(nkpoints_list), nkpoints_list[-1])
        ])
    else:
        nkpoints_list = nkpoints_list[:n_seg]

    # --- 3. Grid lookup tables --------------------------------------------
    kpts_grid = np.asarray(kpts_grid).reshape(-1, 3)
    kpts_wrapped = kpts_grid % 1.0

    kpaths_list = []
    energies_list = []

    # --- 4. Loop over segments --------------------------------------------
    for iseg, seg in enumerate(kpath_segments):
        n_kp = int(nkpoints_list[iseg])

        # Generate dense k-path with ASE
        kp_dense, _, _ = generate_band_path(seg, cell, n_kp)

        n_points = len(kp_dense)

        # Map dense k-points → nearest grid point
        kp_dense_wrapped = kp_dense % 1.0
        e_seg = np.zeros((n_bands, n_points))

        for i_kp, kp in enumerate(kp_dense_wrapped):
            diff = kpts_wrapped - kp
            diff -= np.rint(diff)                 # minimum image for periodicity
            dists = np.linalg.norm(diff, axis=1)
            idx = np.argmin(dists)

            # if dists[idx] > tol:
            #     print(f"Warning: k-point {kp} matched to grid with distance {dists[idx]:.2e}")

            e_seg[:, i_kp] = energies[:, idx]

        kpaths_list.append(kp_dense.copy())
        energies_list.append(e_seg)
        pass


    return kpaths_list, energies_list


    


def extract_kpath_interpolate(
        data, equivalences, coeffs,
    kpath,          # tuple(kpath_string, nkpoints_list) or None for default
    nkpoints_list,
    band_ids=None,
):
    """
    Extract band energies along a high-symmetry k-path from BoltzTraP2 data.

    Uses BoltzTraP2's Fourier interpolation to compute energies at arbitrary
    k-points along the requested path.

    Parameters
    ----------
    data : boltztrap2.BztInterpolatorData
        Object containing lattice vectors, Fermi level, etc.
    equivalences : list
        Symmetry equivalences from ``load_interpolation``.
    coeffs : np.ndarray
        Interpolation coefficients from ``load_interpolation``.
    kpath_list : str or list of str
        Custom k-path. Two format (1) A single str with all k-paths (2) list of k-path segments as str
    nkpoints_list : list[int]
        Number of data points along each k-path
        
    band_ids : list[int] or None, optional
        Subset of band indices to extract. If None, all bands are used.

    Returns
    -------
    kpoints_grid : list[np.ndarray]
        List of dense 3D k-point arrays, one per segment.
        Shape of each array: ``(n_k, 3)``.

    energy_grid : list[np.ndarray]
        List of energy arrays, one per segment.
        Shape of each array: ``(n_bands, n_k)``.

    velocity_grid : 
        Group velocity in [Ha . Bohr] unit.
        Shape of each array: ``(3, n_bands, n_k)``.

    curvature_grid : 
        Band Curvature in [Ha . Bohr^2] unit. Also known as inverse of effective mass.
        Shape of each array: ``(3, 3, n_bands, n_k)``.
    """

    energy_grid = []
    velocity_grid = []
    curvature_grid = []
    kpoints_grid = []

    coeffs_tmp = coeffs
    if band_ids is not None:
        coeffs_tmp = coeffs[band_ids,:]
        pass

    kpaths = parse_k_path(kpath)
    
    for ikpath, kpath in enumerate(kpaths):
        print("k path #{}".format(ikpath + 1))
        # Generate the explicit point list.
        kp, dkp, dcl = generate_band_path(kpath, data.atoms.cell, nkpoints_list[ikpath])
        # print("band path ", kp)
        kpoints_grid.append(kp)
        # Compute the band energies
        with TimerContext() as timer:
            egrid, vgrid, cgrid = fite.getBands(
                kp, equivalences, data.get_lattvec(), coeffs_tmp, curvature=True
            )
            deltat = timer.get_deltat()
            print("rebuilding the bands took {:.3g} s".format(deltat))
        energy_grid.append(egrid)
        velocity_grid.append(vgrid)
        curvature_grid.append(cgrid)
        pass

    return kpoints_grid, energy_grid, velocity_grid, curvature_grid




def plot_and_save_bands_velocity_imass(energy, velocity, curvature, ib, nkpoints_list, labels, efermi):
    fig, axes = plt.subplots(3, len(energy), figsize=(15, 6), sharey="row", gridspec_kw={
        "width_ratios": nkpoints_list,
        "wspace":0,
        "hspace":0
        }, dpi=300)

    for k in range(len(energy)):
        x = np.linspace(0, 1, energy[k].shape[1])
        ax = axes[0, k]
        ax.plot(x, energy[k][ib].T-efermi, label=f"E(K),{ib}")
        ax.axhline(0, 0, 10, color='k', linestyle=":")
        if k == 6:
            ax.legend(framealpha=0.3)
        if k == 0:
            ax.set_ylabel(r"$E-E_F (Ha)$")
        ax.set_xlabel(f"{labels[k]}")
        ax.set_ylim(-0.015, 0.015)


        ax = axes[1, k]
        ax.plot(x, velocity[k][0, ib].T, label="vx")
        ax.plot(x, velocity[k][1, ib].T, label="vy")
        ax.plot(x, velocity[k][2, ib].T, label="vz")
        ax.set_ylim(-0.15, 0.15)

        if k == 6:
            ax.legend(framealpha=0.3)
        if k == 0:
            ax.set_ylabel(r"$v [Ha\cdot Bohr$]")
        ax.set_xlabel(f"{labels[k]}")


        ax = axes[2, k]

        ax.plot(x, curvature[k][0,0, ib].T/100, label="xx")
        ax.plot(x, curvature[k][1,1, ib].T/100, label="yy")
        ax.plot(x, curvature[k][2,2, ib].T/100, label="zz")
        ax.plot(x, curvature[k][0,1, ib].T/100, label="xy")
        ax.plot(x, curvature[k][0,2, ib].T/100, label="xz")
        ax.plot(x, curvature[k][1,2, ib].T/100, label="yz")
        ax.set_ylim(-0.15, 0.15)

        ax.axhline(0, 0, 10, color='k', linestyle=":")
        if k == 6:
            ax.legend(framealpha=0.3)
        if k == 0:
            # ax.set_ylabel(r"$Ha\cdot Bohr^2$")
            ax.set_ylabel(r"$m_e/m^*$")
        ax.set_xlabel(f"{labels[k]}")
        axes[0, k].set_xticks([])
        axes[1, k].set_xticks([])
        axes[2, k].set_xticks([])
        pass
    for ax in axes.flat:
        # Y-axis spine
        ax.spines["left"].set_alpha(0.3)
        ax.spines["right"].set_alpha(0.3)

        # X-axis spine
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["top"].set_linewidth(1.2)

        ax.margins(x=0)           # remove all x-padding
        ax.autoscale(enable=True, axis="x", tight=True)

    plt.savefig(f"Nb3S4-bt2-bands-velocity-curvature-ib{ib}.png")
    pass




def method_1(data):
    """
    using raw data
    """

    kpath_str = (
            "[0.0,0.0,0.0], [0.0,0.0,0.5]"
        )
    nkpoints = 100
    nkpoints_list = np.array([45]) * nkpoints

    kpaths_list, energies_list = extract_kpath_data(data.kpoints, data.ebands, data.atoms.cell,
                                                                    kpath_str, [100],
                                                                    fermi=data.fermi)
    

    k = 0
    x = np.linspace(0, 1, energies_list[k].shape[1])
    plt.plot(x, energies_list[k].T)
    plt.ylim(-0.03, 0.03)




    kpath_str = (
            "[0.0,0.0,0.0], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], "
                "[0.0,0.0,0.0], [0.0,0.0,0.5], [0.5,0.0,0.5], [0.333333,0.333333,0.5], "
                "[0.0,0.0,0.5], [0.5,0.0,0.5], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], [0.333333,0.333333,0.5]"
            )
    labels=["G-M", "M-K", "K-G", "G-A", "A-L", "L-H", "H-A", "A-L", "L-M", "M-K", "K-H"]
    nkpoints = 10
    nkpoints_list = np.array([39, 23, 45, 94, 29, 23, 45, 1, 94, 1, 94, 1]) * nkpoints

    kpaths_list, energies_list = extract_kpath_data(data.kpoints, data.ebands, data.atoms.cell,
                                                                    kpath_str, nkpoints_list,
                                                                    fermi=data.fermi)
    


    k = 3
    x = np.linspace(0, 1, energies_list[k].shape[1])
    plt.plot(x, energies_list[k].T)
    plt.xlabel(f"{labels[k]}")
    plt.legend()
    plt.ylim(-0.04, 0.04)
    

def method_2(data, equivalences, coeffs):
    """
    Using boltztrap interpolation 
    """

    # Testing
    kpath_str = "[0.0,0.0,0.0], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], " \
            "[0.0,0.0,0.0], [0.0,0.0,0.5], [0.5,0.0,0.5], [0.333333,0.333333,0.5], " \
            "[0.0,0.0,0.5], [0.5,0.0,0.5], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], [0.333333,0.333333,0.5]"

    kpoints, energy, velocity, curvature = extract_kpath_interpolate(data, equivalences, coeffs, kpath_str, [100])




    # Testing 2. Each k-path direction is seperated, this way the retured data will be seperated by path as well. 
    # Easier to plot when multiple paths are there
    kpath_str = ["[0.0,0.0,0.0], [0.5, 0.0, 0.0]", 
             "[0.5, 0.0, 0.0], [0.333333,0.333333,0.0]",
             "[0.333333,0.333333,0.0], [0.0,0.0,0.0]",
             "[0.0,0.0,0.0], [0.0,0.0,0.5]",
             "[0.0,0.0,0.5], [0.5,0.0,0.5]", 
             "[0.5,0.0,0.5], [0.333333,0.333333,0.5]", 
             "[0.333333,0.333333,0.5],[0.0,0.0,0.5]",
             "[0.5,0.0,0.5], [0.5, 0.0, 0.0]",
             "[0.333333,0.333333,0.0], [0.333333,0.333333,0.5]"]


    labels=["G-M", "M-K", "K-G", "G-A", "A-L", "L-H", "H-A", "L-M", "K-H"]
    nkpoints = 1
    nkpoints_list = np.array([39, 23, 45, 94, 29, 23, 45, 94, 94]) * nkpoints

    kpoints, energy, velocity, curvature = extract_kpath_interpolate(data, equivalences, coeffs, kpath_str, nkpoints_list)

    plot_and_save_bands_velocity_imass(energy, velocity, curvature, 61, nkpoints_list, labels, data.fermi)
    plot_and_save_bands_velocity_imass(energy, velocity, curvature, 62, nkpoints_list, labels, data.fermi)
    plot_and_save_bands_velocity_imass(energy, velocity, curvature, 63, nkpoints_list, labels, data.fermi)
    plot_and_save_bands_velocity_imass(energy, velocity, curvature, 64, nkpoints_list, labels, data.fermi)

    pass

def main():

    dft_data_dir = "./PBEsol-Relaxed/"
    Efermi_DFT = 0
    niter = 20
    bt2filnam = dft_data_dir + "Nb3S4_BLZTRP_m{}.bt2".format(niter)


    data, equivalences, coeffs = compute.load_interpolation(dft_data_dir, bt2filnam, niter)
    # method_1(data)
    method_2(data, equivalences, coeffs)
    


