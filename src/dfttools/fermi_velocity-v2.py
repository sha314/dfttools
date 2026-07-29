#!/usr/bin/env python3
"""
fermi_velocity.py

Computes Fermi velocity components (vx, vy, vz), their magnitude |v_F|,
and the inverse effective mass tensor 1/m* from a Quantum ESPRESSO pw.x
calculation, using BoltzTraP2's Fourier interpolation.

The band structure is reconstructed on the full uniform k-grid via an
extended version of BoltzTraP2's FFTev function that also computes the
second derivative (curvature) of the bands analytically, giving 1/m*
without any finite differences.

Usage:
    python fermi_velocity.py scf.in
    python fermi_velocity.py scf.in --band-min 10 --band-max 20
    python fermi_velocity.py scf.in --ewin 0.05

Output files (per spin channel, suffix _1/_2 for nspin=2):
    {prefix}_vfermi.frmsf        |v_F| magnitude
    {prefix}_vfermi_x.frmsf      v_x  (signed, Cartesian)
    {prefix}_vfermi_y.frmsf      v_y
    {prefix}_vfermi_z.frmsf      v_z
    {prefix}_inv_mstar.frmsf     Tr(1/m*) = 1/mx + 1/my + 1/mz
    {prefix}_inv_mstar_xx.frmsf  (1/m*)_xx
    {prefix}_inv_mstar_yy.frmsf  (1/m*)_yy
    {prefix}_inv_mstar_zz.frmsf  (1/m*)_zz
    {prefix}_inv_mstar_xy.frmsf  (1/m*)_xy  (off-diagonal)
    {prefix}_inv_mstar_xz.frmsf  (1/m*)_xz
    {prefix}_inv_mstar_yz.frmsf  (1/m*)_yz

Dependencies:
    pip install boltztrap2 ase
"""

import argparse
import os
import sys
import numpy as np
import numpy.fft as npf

# ─── BoltzTraP2 imports ───────────────────────────────────────────────────────
try:
    from BoltzTraP2 import dft, fite, serialization
except ImportError:
    sys.exit(
        "ERROR: BoltzTraP2 not found.\n"
        "Install with:  pip install boltztrap2 ase"
    )

# ─── Constants ────────────────────────────────────────────────────────────────
TPI = 2.0 * np.pi


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def write_frmsf(filename, nk1, nk2, nk3, nbnd, recip_rows, eig, scalar_field):
    """
    Write a FermiSurfer .frmsf file.

    Parameters
    ----------
    nk1, nk2, nk3  : int
    nbnd            : int
    recip_rows      : (3,3) reciprocal lattice vectors as rows, in 2π/alat
    eig             : (nbnd, nk1, nk2, nk3)  energies relative to E_F  [Ry]
    scalar_field    : (nbnd, nk1, nk2, nk3)  quantity to colour the surface
    """
    with open(filename, 'w') as f:
        f.write(f"  {nk1}  {nk2}  {nk3}\n")
        f.write("1\n")
        f.write(f"  {nbnd}\n")
        for vec in recip_rows:
            f.write("  {:18.10f}  {:18.10f}  {:18.10f}\n".format(*vec))
        for ibnd in range(nbnd):
            for i3 in range(nk3):
                for i2 in range(nk2):
                    for i1 in range(nk1):
                        f.write(f"  {eig[ibnd, i1, i2, i3]:18.10f}\n")
            for i3 in range(nk3):
                for i2 in range(nk2):
                    for i1 in range(nk1):
                        f.write(f"  {scalar_field[ibnd, i1, i2, i3]:18.10f}\n")
    print(f"  Written: {filename}")


def read_pw_input(input_file):
    """Parse prefix and outdir from a pw.x input file."""
    prefix = "pwscf"
    outdir = "./"
    with open(input_file) as f:
        for line in f:
            line = line.strip().lower().replace("'", "").replace('"', '')
            if line.startswith("prefix"):
                prefix = line.split("=")[1].strip().rstrip(",")
            elif line.startswith("outdir"):
                outdir = line.split("=")[1].strip().rstrip(",")
    return prefix, outdir


def find_save_dir(prefix, outdir):
    """Return path to the pw.x .save directory."""
    path = os.path.join(outdir, f"{prefix}.save")
    if os.path.isdir(path):
        return path
    raise FileNotFoundError(
        f"Cannot find save directory: {path}\n"
        "Check prefix and outdir in your pw.x input file."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Extended FFT reconstruction: energy + velocity + inverse mass
# ══════════════════════════════════════════════════════════════════════════════

def FFTev2(equivalences, bandcoeff, allvec, dims):
    """
    Reconstruct a single band on the full uniform k-grid via inverse FFT,
    returning the energy, group velocity (first derivative), and inverse
    effective mass tensor (second derivative).

    This extends BoltzTraP2's internal FFTev function by adding the curvature
    (second-derivative) grid using the same Fourier coefficients.

    Physics
    -------
    The band energy is expanded in star functions (symmetrised plane waves):

        ε(k) = Σ_R  c_R · exp(i k·R)

    where R are real-space lattice vectors and c_R are the fitted coefficients.

    First derivative (group velocity):
        ∂ε/∂k_α = Σ_R  c_R · (i R_α) · exp(i k·R)
        → multiply each coefficient by (i · R_α) before IFFT

    Second derivative (inverse effective mass, ℏ=1):
        ∂²ε/∂k_α∂k_β = Σ_R  c_R · (-R_α · R_β) · exp(i k·R)
        → multiply each coefficient by (-R_α · R_β) before IFFT

    The allvec array from BoltzTraP2 already contains the Cartesian k-vectors
    for every member of every equivalence class, so R_α is just allvec[i, α].

    Parameters
    ----------
    equivalences : list of (n_members, 3) int arrays
        k-point equivalence classes in direct (grid) coordinates.
    bandcoeff    : (n_equiv,) complex array
        One Fourier coefficient per equivalence class.
    allvec       : (n_total_kpts, 3) float array
        Cartesian coordinates of every k-point (all members of all classes),
        in units of 2π/alat.  Provided by BoltzTraP2's fite module.
    dims         : (3,) int tuple
        Full k-grid dimensions (nk1, nk2, nk3).

    Returns
    -------
    eb   : (nk1*nk2*nk3,) float   energies on full grid
    vg   : (3, nk1*nk2*nk3) float  group velocities [Ry·bohr]
    inv_m: (3, 3, nk1*nk2*nk3) float  inverse mass tensor [1/ℏ²·Ry/bohr²]
                                       = (1/m*)_αβ in atomic units
    """
    nk_tot = int(np.prod(dims))

    # ── Allocate grids ────────────────────────────────────────────────────────
    egrid  = np.zeros(dims, dtype=complex)
    vgrid  = np.zeros((3,)    + tuple(dims), dtype=complex)   # 1st deriv
    mgrid  = np.zeros((3, 3)  + tuple(dims), dtype=complex)   # 2nd deriv

    # ── Fill grids from Fourier coefficients ──────────────────────────────────
    i = 0
    for coeff, equiv in zip(bandcoeff, equivalences):
        j = i + len(equiv)
        c = coeff / len(equiv)   # normalise by degeneracy (BTP convention)

        ix = equiv[:, 0]
        iy = equiv[:, 1]
        iz = equiv[:, 2]

        # Energy: c · exp(i k·R)  [the exp is implicit in the IFFT]
        egrid[ix, iy, iz] = c

        # Group velocity: i·R_α · c
        kR = allvec[i:j, :]   # (n_members, 3)  Cartesian R vectors
        for a in range(3):
            vgrid[a, ix, iy, iz] = kR[:, a] * c   # factor of i applied below

        # Inverse mass: -R_α·R_β · c
        for a in range(3):
            for b in range(a, 3):
                mgrid[a, b, ix, iy, iz] = -kR[:, a] * kR[:, b] * c

        i = j

    # ── Apply i to velocity (∂/∂k brings down i·R) ───────────────────────────
    vgrid *= 1j

    # ── Inverse FFT on all grids ──────────────────────────────────────────────
    scale = nk_tot   # BTP IFFT convention (see original FFTev)

    eb = scale * npf.ifftn(egrid).real.flatten()

    vg = np.zeros((3, nk_tot))
    for a in range(3):
        vg[a] = scale * npf.ifftn(vgrid[a]).real.flatten()

    inv_m = np.zeros((3, 3, nk_tot))
    for a in range(3):
        for b in range(a, 3):
            val = scale * npf.ifftn(mgrid[a, b]).real.flatten()
            inv_m[a, b] = val
            inv_m[b, a] = val   # tensor is symmetric

    return eb, vg, inv_m


# ══════════════════════════════════════════════════════════════════════════════
# Main computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_bands(data_btp, equivalences, coeffs, dims,
                      band_min=None, band_max=None, ewin=None, ef=0.0):
    """
    Loop over selected bands, call FFTev2 for each, and collect results.

    Parameters
    ----------
    data_btp     : BoltzTraP2 data object (provides .ebands, .mommat, etc.)
    equivalences : from serialization.calc_equivalences
    coeffs       : from fite.fitde3D  shape (nbnd, n_equiv)
    dims         : (nk1, nk2, nk3) full grid dimensions
    band_min/max : 1-based band indices to include (None = all)
    ewin         : energy window [Ry] around E_F; points outside are zeroed
    ef           : Fermi energy [Ry] (already subtracted from eband by BTP,
                   but passed here for the ewin mask)

    Returns
    -------
    eig    : (nbnd_sel, nk1, nk2, nk3)
    vx/vy/vz : (nbnd_sel, nk1, nk2, nk3)  signed velocity components
    vmag   : (nbnd_sel, nk1, nk2, nk3)    |v_F|
    inv_mxx .. inv_myz : (nbnd_sel, nk1, nk2, nk3)  independent tensor elements
    inv_m_trace : (nbnd_sel, nk1, nk2, nk3)  Tr(1/m*)
    """
    nk1, nk2, nk3 = dims
    nk_tot = nk1 * nk2 * nk3
    nbnd_total = coeffs.shape[0]

    b_low  = (band_min - 1) if band_min is not None else 0
    b_high = (band_max - 1) if band_max is not None else nbnd_total - 1
    b_low  = max(0,              b_low)
    b_high = min(nbnd_total - 1, b_high)
    nbnd_sel = b_high - b_low + 1

    print(f"  Bands: {b_low+1} – {b_high+1}  ({nbnd_sel} bands)")

    # BoltzTraP2 stores allvec inside the fite module after fitde3D;
    # we need it for the R-vectors.  Reconstruct from equivalences and
    # the lattice: allvec[i] = R_i in Cartesian (2π/alat units).
    # BTP's fitde3D computes allvec internally — re-derive it the same way.
    allvec = _build_allvec(equivalences, data_btp.get_lattvec())

    shape = (nbnd_sel, nk1, nk2, nk3)
    eig         = np.zeros(shape)
    vx          = np.zeros(shape)
    vy          = np.zeros(shape)
    vz          = np.zeros(shape)
    vmag        = np.zeros(shape)
    inv_mxx     = np.zeros(shape)
    inv_myy     = np.zeros(shape)
    inv_mzz     = np.zeros(shape)
    inv_mxy     = np.zeros(shape)
    inv_mxz     = np.zeros(shape)
    inv_myz     = np.zeros(shape)
    inv_m_trace = np.zeros(shape)

    for ib, iband in enumerate(range(b_low, b_high + 1)):
        print(f"    Band {iband+1}/{nbnd_total} ...", end="\r")
        eb, vg, inv_m = FFTev2(equivalences, coeffs[iband], allvec, dims)

        eig[ib]     = eb.reshape(nk1, nk2, nk3)
        vx[ib]      = vg[0].reshape(nk1, nk2, nk3)
        vy[ib]      = vg[1].reshape(nk1, nk2, nk3)
        vz[ib]      = vg[2].reshape(nk1, nk2, nk3)
        vmag[ib]    = np.sqrt(vg[0]**2 + vg[1]**2 + vg[2]**2).reshape(nk1, nk2, nk3)
        inv_mxx[ib] = inv_m[0, 0].reshape(nk1, nk2, nk3)
        inv_myy[ib] = inv_m[1, 1].reshape(nk1, nk2, nk3)
        inv_mzz[ib] = inv_m[2, 2].reshape(nk1, nk2, nk3)
        inv_mxy[ib] = inv_m[0, 1].reshape(nk1, nk2, nk3)
        inv_mxz[ib] = inv_m[0, 2].reshape(nk1, nk2, nk3)
        inv_myz[ib] = inv_m[1, 2].reshape(nk1, nk2, nk3)
        inv_m_trace[ib] = (inv_m[0,0] + inv_m[1,1] + inv_m[2,2]).reshape(nk1, nk2, nk3)

    print()   # newline after progress

    # ── Energy window mask ────────────────────────────────────────────────────
    if ewin is not None:
        mask = np.abs(eig) > ewin
        for arr in (vmag, vx, vy, vz,
                    inv_mxx, inv_myy, inv_mzz,
                    inv_mxy, inv_mxz, inv_myz, inv_m_trace):
            arr[mask] = 0.0

    return (eig, vmag, vx, vy, vz,
            inv_mxx, inv_myy, inv_mzz,
            inv_mxy, inv_mxz, inv_myz,
            inv_m_trace,
            b_low, b_high)


def _build_allvec(equivalences, lattvec):
    """
    Reconstruct the allvec array (Cartesian R-vectors for every k-point
    in every equivalence class) from the equivalence list and the real-space
    lattice vectors.

    BoltzTraP2's fite.fitde3D builds this internally but does not expose it.
    We replicate the same construction:
        R_cart = lattvec @ n    where n = (n1, n2, n3) are grid indices
    stored as floats in 2π/alat Cartesian coordinates.

    lattvec : (3,3)  real-space lattice vectors as columns [bohr]
              (as returned by data.get_lattvec())
    """
    rows = []
    for equiv in equivalences:
        # equiv : (n_members, 3) integer grid coordinates
        # R_cart = lattvec @ n.T  → shape (3, n_members)
        R = lattvec @ equiv.T   # (3, n_members)
        rows.append(R.T)        # (n_members, 3)
    return np.vstack(rows)      # (n_total, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Fermi velocity and inverse effective mass from QE output "
            "using BoltzTraP2 Fourier interpolation."
        )
    )
    parser.add_argument("input_file",
                        help="pw.x input file (contains prefix and outdir)")
    parser.add_argument("--band-min", type=int, default=None,
                        help="First band to include, 1-based (default: all)")
    parser.add_argument("--band-max", type=int, default=None,
                        help="Last band to include, 1-based (default: all)")
    parser.add_argument("--ewin", type=float, default=None,
                        help="Energy window [Ry] around E_F; outside → zero")
    args = parser.parse_args()

    # ── Locate QE data ────────────────────────────────────────────────────────
    print(f"Reading pw.x input: {args.input_file}")
    prefix, outdir = read_pw_input(args.input_file)
    save_dir = find_save_dir(prefix, outdir)
    print(f"  prefix={prefix}  outdir={outdir}")

    # ── Load via BoltzTraP2 ───────────────────────────────────────────────────
    print("Loading QE data via BoltzTraP2 ...")
    data_btp = dft.QuantumEspressoData(save_dir, derivatives=False)
    #   data_btp.ebands  : (nspin, nbnd, nk_irr)  [Ry, relative to E_F]
    #   data_btp.kpoints : (nk_irr, 3)  in reduced coordinates
    #   data_btp.atoms   : ASE Atoms object

    nspin   = data_btp.ebands.shape[0]
    nbnd    = data_btp.ebands.shape[1]
    nk_irr  = data_btp.ebands.shape[2]

    # k-grid dimensions from the data object
    # BTP exposes dims as data_btp.mommat or via equivalences
    equivalences = serialization.calc_equivalences(data_btp)
    # dims = upper bound on k-grid indices, inferred from equivalences
    dims = tuple(
        int(np.max([eq[:, i].max() for eq in equivalences]) + 1)
        for i in range(3)
    )
    nk1, nk2, nk3 = dims
    print(f"  k-grid: {nk1}×{nk2}×{nk3}   nk_irr={nk_irr}   nbnd={nbnd}   nspin={nspin}")

    # Reciprocal lattice rows for frmsf header
    # data_btp.get_lattvec() returns real-space vectors; reciprocal = 2π (lattvec^{-T})
    lattvec = data_btp.get_lattvec()                # (3,3) columns, bohr
    recip   = TPI * np.linalg.inv(lattvec).T        # (3,3) columns, 1/bohr
    recip_rows = recip.T                             # rows for frmsf

    base = os.path.join(outdir, prefix)
    spin_labels = [f"_{s+1}" for s in range(nspin)] if nspin > 1 else [""]

    for ispin in range(nspin):
        slabel = spin_labels[ispin]

        # ── Fit Fourier coefficients for this spin channel ────────────────────
        print(f"\nFitting Fourier coefficients (spin {ispin+1}/{nspin}) ...")
        # ebands for this spin: (nbnd, nk_irr)
        ebands_spin = data_btp.ebands[ispin]   # already shifted to E_F by BTP
        coeffs = fite.fitde3D(data_btp, equivalences)
        # coeffs shape: (nbnd, n_equiv)

        # ── Reconstruct on full grid ──────────────────────────────────────────
        print("Reconstructing bands and derivatives on full k-grid ...")
        (eig, vmag, vx, vy, vz,
         inv_mxx, inv_myy, inv_mzz,
         inv_mxy, inv_mxz, inv_myz,
         inv_m_trace,
         b_low, b_high) = compute_all_bands(
            data_btp, equivalences, coeffs, dims,
            band_min=args.band_min,
            band_max=args.band_max,
            ewin=args.ewin,
        )
        nbnd_sel = b_high - b_low + 1

        # ── Write .frmsf files ────────────────────────────────────────────────
        print("Writing FermiSurfer files ...")
        quantities = {
            "vfermi"       : vmag,
            "vfermi_x"     : vx,
            "vfermi_y"     : vy,
            "vfermi_z"     : vz,
            "inv_mstar"    : inv_m_trace,
            "inv_mstar_xx" : inv_mxx,
            "inv_mstar_yy" : inv_myy,
            "inv_mstar_zz" : inv_mzz,
            "inv_mstar_xy" : inv_mxy,
            "inv_mstar_xz" : inv_mxz,
            "inv_mstar_yz" : inv_myz,
        }
        for qname, qarr in quantities.items():
            fname = f"{base}_{qname}{slabel}.frmsf"
            write_frmsf(fname, nk1, nk2, nk3, nbnd_sel,
                        recip_rows, eig, qarr)

    print("\nDone.")
    print("Example FermiSurfer usage:")
    print(f"  fermisurfer {prefix}_vfermi.frmsf")
    print(f"  fermisurfer {prefix}_vfermi_x.frmsf")
    print(f"  fermisurfer {prefix}_inv_mstar.frmsf")
    print(f"  fermisurfer {prefix}_inv_mstar_xx.frmsf")


if __name__ == "__main__":
    main()