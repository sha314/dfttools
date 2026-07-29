#!/usr/bin/env python3
"""
fermi_velocity.py

Python equivalent of the Quantum ESPRESSO fermi_velocity.x post-processing tool.

Usage:
    python fermi_velocity.py <pw.x input file>

Outputs (per spin channel):
    {prefix}_vfermi.frmsf      - |v_F| magnitude (FermiSurfer format)
    {prefix}_vfermi_x.frmsf   - v_x component
    {prefix}_vfermi_y.frmsf   - v_y component
    {prefix}_vfermi_z.frmsf   - v_z component

For nspin=2, each file is duplicated with suffixes _1 and _2 for spin-up/down.

Modifications vs. original Fortran code:
  - Outputs vx, vy, vz components in addition to |v| magnitude
  - Reads QE XML output directly via the `qeschema` library (or via manual XML parsing)
  - Pure Python/NumPy: no MPI, no Fortran runtime required
  - Command-line interface via argparse
  - Optional band range filtering (--band-min, --band-max)
  - Optional energy window filtering (--ewin) around E_F
"""

import argparse
import os
import sys
import struct
import numpy as np

# ─── Optional: use qeschema for reading QE XML ────────────────────────────────
# Install with: pip install qeschema
# If not available, we fall back to a lightweight manual XML reader.
try:
    import qeschema
    HAS_QESCHEMA = True
except ImportError:
    HAS_QESCHEMA = False

# ─── Constants ────────────────────────────────────────────────────────────────
TPI = 2.0 * np.pi          # 2π
BOHR_TO_ANGSTROM = 0.529177210903


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def write_frmsf(filename, nk1, nk2, nk3, nbnd, lattice_vectors, eig, scalar_field):
    """
    Write a FermiSurfer .frmsf file.

    Parameters
    ----------
    filename        : str
    nk1,nk2,nk3    : int   - k-grid dimensions
    nbnd            : int   - number of bands written
    lattice_vectors : (3,3) array in Bohr  (reciprocal, Cartesian, 2π/alat units)
    eig             : (nbnd, nk1, nk2, nk3)  eigenvalues relative to E_F  [Ry]
    scalar_field    : (nbnd, nk1, nk2, nk3)  quantity to colour the surface
    """
    with open(filename, 'w') as f:
        # Header: k-grid
        f.write(f"  {nk1}  {nk2}  {nk3}\n")
        # 1 = number of Fermi surfaces written per band crossing (always 1 here)
        f.write("1\n")
        f.write(f"  {nbnd}\n")
        # Reciprocal lattice vectors (rows), in units of 2π/alat
        for vec in lattice_vectors:
            f.write("  {:18.10f}  {:18.10f}  {:18.10f}\n".format(*vec))
        # Eigenvalues then scalar quantity, band by band, k-point by k-point
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


# ══════════════════════════════════════════════════════════════════════════════
# QE XML reader (lightweight fallback if qeschema not installed)
# ══════════════════════════════════════════════════════════════════════════════

def find_save_dir(prefix, outdir):
    """Return path to the pw.x save directory."""
    candidates = [
        os.path.join(outdir, f"{prefix}.save"),
        os.path.join(outdir, f"{prefix}.save/"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"Cannot find save directory for prefix='{prefix}' in outdir='{outdir}'.\n"
        f"Tried: {candidates}"
    )


def read_pw_input(input_file):
    """
    Minimal parser for the pw.x input file.
    Returns (prefix, outdir) as strings.
    """
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


def read_qe_xml(save_dir):
    """
    Read the data-file-schema.xml produced by QE ≥ 6.4.
    Returns a dict with keys:
        alat, at (3x3), bg (3x3), nspin,
        nks, nk1, nk2, nk3,
        et (nbnd x nks),  [Ry]
        ef, ef_up, ef_dw,  [Ry]
        two_fermi_energies,
        nbnd
    """
    import xml.etree.ElementTree as ET

    xml_path = os.path.join(save_dir, "data-file-schema.xml")
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Cannot find {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    def txt(node, tag, default=None):
        el = node.find(tag)
        if el is None:
            return default
        return el.text.strip()

    def ftxt(node, tag, default=0.0):
        v = txt(node, tag)
        return float(v) if v is not None else default

    # ── cell ──────────────────────────────────────────────────────────────────
    cell_node = root.find(".//output/atomic_structure")
    alat = float(cell_node.attrib.get("alat", 1.0))

    cell = root.find(".//output/atomic_structure/cell")
    a1 = np.array(cell.find("a1").text.split(), dtype=float)
    a2 = np.array(cell.find("a2").text.split(), dtype=float)
    a3 = np.array(cell.find("a3").text.split(), dtype=float)
    # at columns = lattice vectors in Cartesian (Bohr / alat)
    at = np.column_stack([a1, a2, a3]) / alat

    # reciprocal cell
    bg_node = root.find(".//output/basis_set/reciprocal_lattice")
    b1 = np.array(bg_node.find("b1").text.split(), dtype=float)
    b2 = np.array(bg_node.find("b2").text.split(), dtype=float)
    b3 = np.array(bg_node.find("b3").text.split(), dtype=float)
    bg = np.column_stack([b1, b2, b3])

    # ── spin ──────────────────────────────────────────────────────────────────
    mag_node = root.find(".//output/magnetization")
    nspin = int(txt(mag_node, "noncolin", "false") == "true" and 4 or
                txt(mag_node, "lsda", "false") == "true" and 2 or 1)
    two_fermi = txt(mag_node, "two_fermi_energies", "false").lower() == "true"

    # ── Fermi energy ─────────────────────────────────────────────────────────
    band_struct = root.find(".//output/band_structure")
    ef   = ftxt(band_struct, "fermi_energy")      # Ry
    ef_up = ftxt(band_struct, "two_fermi_energies/fermi_energy_up", ef)
    ef_dw = ftxt(band_struct, "two_fermi_energies/fermi_energy_dw", ef)

    # ── k-mesh ────────────────────────────────────────────────────────────────
    nks_node = band_struct.find("nks")
    nks = int(nks_node.text)

    # Try to read nk1/nk2/nk3 from monkhorst_pack element
    mp = root.find(".//input/k_points_IBZ/monkhorst_pack")
    if mp is not None:
        nk1 = int(mp.attrib["nk1"])
        nk2 = int(mp.attrib["nk2"])
        nk3 = int(mp.attrib["nk3"])
    else:
        # fallback: read from start_k info in output (if present)
        mp2 = root.find(".//output/symmetries/nk1")
        if mp2 is not None:
            nk1 = int(root.find(".//output/symmetries/nk1").text)
            nk2 = int(root.find(".//output/symmetries/nk2").text)
            nk3 = int(root.find(".//output/symmetries/nk3").text)
        else:
            raise RuntimeError(
                "Cannot determine k-grid dimensions from XML. "
                "Make sure you used a Monkhorst-Pack mesh in pw.x."
            )

    # ── eigenvalues ───────────────────────────────────────────────────────────
    nbnd = int(band_struct.find("nbnd").text)
    ks_energies = band_struct.findall("ks_energies")

    et = np.zeros((nbnd, nks))
    for ik, ks in enumerate(ks_energies):
        evals_text = ks.find("eigenvalues").text.split()
        et[:, ik] = [float(x) for x in evals_text]  # already in Ry

    return dict(
        alat=alat, at=at, bg=bg,
        nspin=nspin, two_fermi_energies=two_fermi,
        nks=nks, nk1=nk1, nk2=nk2, nk3=nk3,
        nbnd=nbnd, et=et,
        ef=ef, ef_up=ef_up, ef_dw=ef_dw,
    )


# ══════════════════════════════════════════════════════════════════════════════
# equiv-k mapping (reproduces rotate_k_fs from Fortran)
# ══════════════════════════════════════════════════════════════════════════════

def build_equiv_map(save_dir, nk1, nk2, nk3, nks_half, nspin):
    """
    Build equiv[i1,i2,i3] = 0-based index into et[:,ik] for the irreducible BZ.

    We read the k-point coordinates from the XML and map each full-BZ point
    to its irreducible representative by symmetry.

    This is a simplified version: it reads all k-points from the XML, builds
    a lookup table in fractional coordinates, and for each full-BZ grid point
    searches (with modular arithmetic) for a matching irreducible k-point.
    """
    import xml.etree.ElementTree as ET

    xml_path = os.path.join(save_dir, "data-file-schema.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    band_struct = root.find(".//output/band_structure")
    ks_list = band_struct.findall("ks_energies")

    # Read irreducible k-points in crystal (fractional) coordinates
    k_irr = []
    for ks in ks_list[:nks_half]:   # first nks/nspin entries = spin-up or non-spin
        kpt_text = ks.find("k_point").text.split()
        k_irr.append([float(x) for x in kpt_text])   # in 2π/alat Cartesian

    # We need fractional coords → convert from Cartesian (2π/alat) to crystal
    # bg rows = reciprocal vectors in 2π/alat Cartesian
    bg_node = root.find(".//output/basis_set/reciprocal_lattice")
    b1 = np.array(bg_node.find("b1").text.split(), dtype=float)
    b2 = np.array(bg_node.find("b2").text.split(), dtype=float)
    b3 = np.array(bg_node.find("b3").text.split(), dtype=float)
    bg = np.column_stack([b1, b2, b3])   # (3,3), columns = b1,b2,b3

    # fractional = bg^{-T} @ k_cart
    inv_bg = np.linalg.inv(bg.T)
    k_irr_frac = np.array([(inv_bg @ np.array(k)) for k in k_irr])
    k_irr_frac = np.mod(k_irr_frac, 1.0)   # fold to [0,1)

    equiv = np.zeros((nk1, nk2, nk3), dtype=int)
    tol = 1.0 / max(nk1, nk2, nk3) * 0.5

    sym_ops = read_symmetries(root, bg)
    for i1 in range(nk1):
        for i2 in range(nk2):
            for i3 in range(nk3):
                kfrac = np.array([i1 / nk1, i2 / nk2, i3 / nk3])
                # Try kfrac and symmetry-equivalent images
                found = False
                for ik, kirr in enumerate(k_irr_frac):
                    diff = np.mod(kfrac - kirr + 0.5, 1.0) - 0.5
                    if np.max(np.abs(diff)) < tol:
                        equiv[i1, i2, i3] = ik
                        found = True
                        break
                if not found:
                    # Try all symmetry operations from XML
                    # sym_ops = read_symmetries(root, bg)
                    for rot, _ftrans in sym_ops:
                        k_sym = np.mod(rot.T @ kfrac, 1.0)
                        for ik, kirr in enumerate(k_irr_frac):
                            diff = np.mod(k_sym - kirr + 0.5, 1.0) - 0.5
                            if np.max(np.abs(diff)) < tol:
                                equiv[i1, i2, i3] = ik
                                found = True
                                break
                        if found:
                            break
                if not found:
                    raise RuntimeError(
                        f"No irreducible k-point found for grid point "
                        f"({i1}/{nk1}, {i2}/{nk2}, {i3}/{nk3}). "
                        "Check that the k-grid in pw.x matches the calculation."
                    )
    return equiv


# def read_symmetries(root, bg):
#     """Read rotation matrices from XML and return list of (3,3) arrays in crystal coords."""
#     syms = []
#     for sym in root.findall(".//output/symmetries/symmetry/rotation"):
#         rows = sym.text.strip().split("\n")
#         mat = np.array([[float(x) for x in row.split()] for row in rows])
#         syms.append(mat)
#         pass
#     print(f"There are {len(syms)} symmetry operations")
#     return syms


def read_symmetries(root, bg):
    """
    Read all symmetry operations from QE XML.
 
    Returns a list of (rotation, fractional_translation) pairs where:
      - rotation            : (3,3) integer matrix in crystal coordinates
      - fractional_translation : (3,) float vector in crystal coordinates
 
    QE stores rotations in crystal coordinates (integer entries ±1, 0) and
    fractional translations in Cartesian units of 2π/alat.  We convert the
    translation to crystal coordinates using bg^{-T}.
 
    For symmorphic space groups every translation is zero.
    For non-symmorphic space groups (glide planes, screw axes) the translations
    are non-zero (typically 0.5 in one or more crystal directions).
 
    The action of operation s on a reciprocal-space vector k (crystal coords) is:
        k' = R^T · k          (translations vanish in k-space for Bravais lattice)
    but when *unfolding* the IBZ back to the full BZ we apply the rotation to
    real-space fractional coordinates, so we keep translations for completeness
    and correctness in edge cases.
 
    Note: in reciprocal space, fractional translations only contribute a phase
    to the Bloch wave function and do NOT change the k-point itself.  So for
    the purpose of building equiv[] we only need the rotation part.  This
    function still returns translations so callers can use them if needed.
    """
    sym_ops = []
    for sym_node in root.findall(".//output/symmetries/symmetry"):
        rot_node = sym_node.find("rotation")
        if rot_node is None:
            continue
        rows = rot_node.text.strip().split("\n")
        rot = np.array([[float(x) for x in row.split()] for row in rows])
 
        # Fractional translation (may be absent for symmorphic ops)
        ftrans = np.zeros(3)
        ft_node = sym_node.find("fractional_translation")
        if ft_node is not None:
            # QE writes it in Cartesian 2π/alat units
            ft_cart = np.array(ft_node.text.split(), dtype=float)
            # Convert to crystal coordinates: f_cryst = bg^{-T} · f_cart
            inv_bgT = np.linalg.inv(bg.T)
            ftrans = inv_bgT @ ft_cart
 
        sym_ops.append((rot, ftrans))
 
    if not sym_ops:
        # Fallback: identity only (shouldn't happen with a proper QE XML)
        sym_ops = [(np.eye(3, dtype=float), np.zeros(3))]
    print(f"There are {len(sym_ops)} symmetry operations")
    return sym_ops

# ══════════════════════════════════════════════════════════════════════════════
# Main computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_fermi_velocity(data, equiv, band_min=None, band_max=None, ewin=None):
    """
    Compute Fermi velocity components and magnitude over the full BZ.

    Returns
    -------
    eig  : (nbnd_sel, nk1, nk2, nk3, ns)  eigenvalues - E_F  [Ry]
    vf   : (nbnd_sel, nk1, nk2, nk3, ns)  |v_F|
    vx   : (nbnd_sel, nk1, nk2, nk3, ns)  v_x (Cartesian, atomic units)
    vy   : ...
    vz   : ...
    b_low, b_high : int  0-based band indices selected
    """
    at   = data["at"]          # (3,3) direct lattice vectors (columns), in alat
    alat = data["alat"]        # Bohr
    nk1, nk2, nk3 = data["nk1"], data["nk2"], data["nk3"]
    nspin = data["nspin"]
    ns = 2 if nspin == 2 else 1
    nk = data["nks"] // ns
    nbnd = data["nbnd"]
    et   = data["et"]          # (nbnd, nks)  Ry

    ef   = data["ef"]
    ef_up = data["ef_up"] if data["two_fermi_energies"] else ef
    ef_dw = data["ef_dw"] if data["two_fermi_energies"] else ef
    efermi = [ef_up, ef_dw] if ns == 2 else [ef]

    # ── Band selection ────────────────────────────────────────────────────────
    b_low  = (band_min - 1) if band_min is not None else 0
    b_high = (band_max - 1) if band_max is not None else nbnd - 1
    b_low  = max(0,       b_low)
    b_high = min(nbnd-1,  b_high)
    nbnd_sel = b_high - b_low + 1

    # ── Map eigenvalues to full BZ ────────────────────────────────────────────
    # eig shape: (nbnd_sel, nk1, nk2, nk3, ns)
    eig = np.zeros((nbnd_sel, nk1, nk2, nk3, ns))

    for ispin in range(ns):
        ef_s = efermi[ispin]
        offset = ispin * nk
        for i3 in range(nk3):
            for i2 in range(nk2):
                for i1 in range(nk1):
                    ik_irr = equiv[i1, i2, i3]
                    eig[:, i1, i2, i3, ispin] = (
                        et[b_low:b_high+1, ik_irr + offset] - ef_s
                    )

    # ── Fermi velocity via finite differences ─────────────────────────────────
    # v_n(k) = ∇_k ε_n(k) = (1/ℏ) ∂ε/∂k
    # In atomic units ℏ=1; ε in Ry; k in 2π/alat  →  v in alat·Ry
    # Central difference: dε/dk_i ≈ [ε(k+δk_i) - ε(k-δk_i)] / 2δk_i
    #                               = [ε+ - ε-] * (nk_i/2)   [crystal units]
    # Then rotate to Cartesian: v_cart = at^T · v_cryst  (at in alat)
    # Multiply by alat/tpi to convert from 1/(2π/alat) to alat units
    # Final units: Ry·a_0 (atomic units of velocity * ℏ, standard for v_F)

    nk_vec = np.array([nk1, nk2, nk3], dtype=float)

    vx = np.zeros_like(eig)
    vy = np.zeros_like(eig)
    vz = np.zeros_like(eig)
    vf = np.zeros_like(eig)

    # Use vectorised numpy roll for speed
    for ispin in range(ns):
        eig_s = eig[:, :, :, :, ispin]   # (nbnd_sel, nk1, nk2, nk3)

        de = np.zeros((3,) + eig_s.shape)
        for ii in range(3):
            # roll axis ii+1 (band is axis 0, k-axes are 1,2,3)
            ep = np.roll(eig_s, -1, axis=ii+1)
            em = np.roll(eig_s,  1, axis=ii+1)
            de[ii] = (ep - em) * 0.5 * nk_vec[ii]   # crystal coord gradient

        # Rotate to Cartesian: de_cart_α = Σ_i at[α,i] * de[i]
        # at is (3,3) with columns = lattice vectors → at[row=α, col=i]
        de_x = sum(at[0, i] * de[i] for i in range(3)) * alat / TPI
        de_y = sum(at[1, i] * de[i] for i in range(3)) * alat / TPI
        de_z = sum(at[2, i] * de[i] for i in range(3)) * alat / TPI

        vx[:, :, :, :, ispin] = de_x
        vy[:, :, :, :, ispin] = de_y
        vz[:, :, :, :, ispin] = de_z
        vf[:, :, :, :, ispin] = np.sqrt(de_x**2 + de_y**2 + de_z**2)

    # ── Optional energy window filter ─────────────────────────────────────────
    if ewin is not None:
        mask = np.abs(eig) > ewin    # far from E_F → zero out (keeps frmsf valid)
        vf[mask] = 0.0
        vx[mask] = 0.0
        vy[mask] = 0.0
        vz[mask] = 0.0

    return eig, vf, vx, vy, vz, b_low, b_high


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute Fermi velocity from QE pw.x output and write FermiSurfer files."
    )
    parser.add_argument("input_file", help="pw.x input file (contains prefix and outdir)")
    parser.add_argument("--band-min", type=int, default=None,
                        help="First band index to include (1-based, default: all)")
    parser.add_argument("--band-max", type=int, default=None,
                        help="Last band index to include (1-based, default: all)")
    parser.add_argument("--ewin", type=float, default=None,
                        help="Energy window [Ry] around E_F; k-points outside are zeroed")
    args = parser.parse_args()

    # ── Read input ────────────────────────────────────────────────────────────
    print(f"Reading pw.x input file: {args.input_file}")
    prefix, outdir = read_pw_input(args.input_file)
    print(f"  prefix = {prefix}")
    print(f"  outdir = {outdir}")

    save_dir = find_save_dir(prefix, outdir)
    print(f"  save_dir = {save_dir}")

    print("Reading QE XML data...")
    data = read_qe_xml(save_dir)

    nk1, nk2, nk3 = data["nk1"], data["nk2"], data["nk3"]
    nspin = data["nspin"]
    ns = 2 if nspin == 2 else 1
    nk = data["nks"] // ns

    print(f"  k-grid: {nk1} x {nk2} x {nk3}  (nks_irr={nk}, nspin={nspin})")
    print(f"  nbnd = {data['nbnd']}")
    print(f"  E_F  = {data['ef']:.6f} Ry")
    if data["two_fermi_energies"]:
        print(f"  E_F↑ = {data['ef_up']:.6f} Ry")
        print(f"  E_F↓ = {data['ef_dw']:.6f} Ry")

    # ── Build equiv map ───────────────────────────────────────────────────────
    print("Building irreducible→full BZ mapping...")
    equiv = build_equiv_map(save_dir, nk1, nk2, nk3, nk, nspin)

    # ── Compute velocities ────────────────────────────────────────────────────
    print("Computing Fermi velocities...")
    eig, vf, vx, vy, vz, b_low, b_high = compute_fermi_velocity(
        data, equiv,
        band_min=args.band_min,
        band_max=args.band_max,
        ewin=args.ewin,
    )
    nbnd_sel = b_high - b_low + 1
    print(f"  Bands included: {b_low+1} - {b_high+1}  ({nbnd_sel} bands)")

    # ── Reciprocal lattice for frmsf header ──────────────────────────────────
    bg = data["bg"]   # (3,3) columns = b1,b2,b3  in 2π/alat
    # frmsf expects rows = reciprocal vectors
    recip_rows = bg.T   # (3,3) rows

    # ── Write FermiSurfer files ───────────────────────────────────────────────
    base = os.path.join(outdir, prefix)

    spin_labels = ["1", "2"] if ns == 2 else [""]

    quantities = {
        "vfermi":   vf,
        "vfermi_x": vx,
        "vfermi_y": vy,
        "vfermi_z": vz,
    }

    print("Writing FermiSurfer files...")
    for ispin, slabel in enumerate(spin_labels):
        suffix = f"_{slabel}" if slabel else ""
        for qname, qarr in quantities.items():
            fname = f"{base}_{qname}{suffix}.frmsf"
            write_frmsf(
                fname,
                nk1, nk2, nk3, nbnd_sel,
                recip_rows,
                eig[:, :, :, :, ispin],
                qarr[:, :, :, :, ispin],
            )

    print("\nDone. Load the .frmsf files in FermiSurfer:")
    print("  fermisurfer <prefix>_vfermi.frmsf       # colour by |v_F|")
    print("  fermisurfer <prefix>_vfermi_x.frmsf     # colour by v_x")
    print("  fermisurfer <prefix>_vfermi_y.frmsf     # colour by v_y")
    print("  fermisurfer <prefix>_vfermi_z.frmsf     # colour by v_z")




def main_v2(save_dir):
    print("Reading QE XML data...")
    data = read_qe_xml(save_dir)

    nk1, nk2, nk3 = data["nk1"], data["nk2"], data["nk3"]
    nspin = data["nspin"]
    ns = 2 if nspin == 2 else 1
    nk = data["nks"] // ns

    print(f"  k-grid: {nk1} x {nk2} x {nk3}  (nks_irr={nk}, nspin={nspin})")
    print(f"  nbnd = {data['nbnd']}")
    print(f"  E_F  = {data['ef']:.6f} Ry")
    if data["two_fermi_energies"]:
        print(f"  E_F↑ = {data['ef_up']:.6f} Ry")
        print(f"  E_F↓ = {data['ef_dw']:.6f} Ry")

    # ── Build equiv map ───────────────────────────────────────────────────────
    print("Building irreducible→full BZ mapping...")
    equiv = build_equiv_map(save_dir, nk1, nk2, nk3, nk, nspin)

    # ── Compute velocities ────────────────────────────────────────────────────
    print("Computing Fermi velocities...")
    eig, vf, vx, vy, vz, b_low, b_high = compute_fermi_velocity(
        data, equiv,
        band_min=None,
        band_max=None,
        ewin=None,
    )
    nbnd_sel = b_high - b_low + 1
    print(f"  Bands included: {b_low+1} - {b_high+1}  ({nbnd_sel} bands)")

    # ── Reciprocal lattice for frmsf header ──────────────────────────────────
    bg = data["bg"]   # (3,3) columns = b1,b2,b3  in 2π/alat
    # frmsf expects rows = reciprocal vectors
    recip_rows = bg.T   # (3,3) rows

    # ── Write FermiSurfer files ───────────────────────────────────────────────
    base = os.path.join(outdir, prefix)

    spin_labels = ["1", "2"] if ns == 2 else [""]

    quantities = {
        "vfermi":   vf,
        "vfermi_x": vx,
        "vfermi_y": vy,
        "vfermi_z": vz,
    }

    print("Writing FermiSurfer files...")
    for ispin, slabel in enumerate(spin_labels):
        suffix = f"_{slabel}" if slabel else ""
        for qname, qarr in quantities.items():
            fname = f"{base}_{qname}{suffix}.frmsf"
            write_frmsf(
                fname,
                nk1, nk2, nk3, nbnd_sel,
                recip_rows,
                eig[:, :, :, :, ispin],
                qarr[:, :, :, :, ispin],
            )

    print("\nDone. Load the .frmsf files in FermiSurfer:")
    print("  fermisurfer <prefix>_vfermi.frmsf       # colour by |v_F|")
    print("  fermisurfer <prefix>_vfermi_x.frmsf     # colour by v_x")
    print("  fermisurfer <prefix>_vfermi_y.frmsf     # colour by v_y")
    print("  fermisurfer <prefix>_vfermi_z.frmsf     # colour by v_z")


if __name__ == "__main__":
    # main()
    main_v2("/Users/shahnoor/projects/materials-project/dft-nb3s4-data-22x22x54-our-param/PBEsol-Relaxed")