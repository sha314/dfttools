# This script takes data-file-schema.xml as input and creates one or more 
# .energy and .structure files with selected bands for boltztrap to interpolate
# It will be useful for energy resolved transport computation

# You can specify energy range and it will take the bands in that range only to create .energy files
# Or you can specify band index and it will only write that file
# For band resolved transport coefficients, I think it's better to first find the bands in an energy range
# and then create sepearte .energy files for those bands


import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
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
import matplotlib.pyplot as plt
import ast
import itertools

import matplotlib
import ase.dft.kpoints as asekp

from BoltzTraP2.misc import TimerContext, dir_context, info, lexit, warning



Ha_to_eV = 27.211396132
Ry_to_eV = 13.6057039763
Ry_to_Ha = Ry_to_eV/Ha_to_eV
Energy_Unit_Conv = 2.0

def load_interpolation(dft_data_dir, bt2filnam, niter=1):
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



# you need the reciprocal lattice vectors from the XML
# find them under: output > basis_set > reciprocal_lattice
# b1, b2, b3 in units of 2π/a

def get_reciprocal_lattice(root, ns):
    rl = root.find(".//reciprocal_lattice", ns)
    b1 = np.array(rl.find("b1").text.split(), dtype=float)
    b2 = np.array(rl.find("b2").text.split(), dtype=float)
    b3 = np.array(rl.find("b3").text.split(), dtype=float)
    return np.array([b1, b2, b3])   # shape (3,3), rows are b1,b2,b3

def parse_qe_xml(filepath):
    """
        All energies are in Ry in the data-file-schema file and it will be kept in Ry
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    # namespace — QE wraps everything in a namespace
    ns = {}
    tag = root.tag
    if tag.startswith("{"):
        ns_url = tag[1:tag.index("}")]
        ns = {"qes": ns_url}
        def find(node, path):
            return node.find(path, ns)
        def findall(node, path):
            return node.findall(path, ns)
    else:
        def find(node, path):
            # strip namespace prefixes for plain XML
            return node.find(path)
        def findall(node, path):
            return node.findall(path)

    band_structure = find(root, ".//band_structure")

    nbnd = int(find(band_structure, "nbnd").text.strip())
    nks  = int(find(band_structure, "nks").text.strip())
    ef   = float(find(band_structure, "fermi_energy").text.strip()) * Energy_Unit_Conv


    B = get_reciprocal_lattice(root, ns)   # reciprocal lattice matrix



    print(f"nbnd = {nbnd}, nks = {nks}, E_fermi = {ef:.4f} eV")

    records = []
    for ik, ks in enumerate(findall(band_structure, "ks_energies")):
        kpt    = find(ks, "k_point")
        weight = float(kpt.attrib.get("weight", 0.0))
        kx, ky, kz = map(float, kpt.text.split())

        # print(kx, ky, kz)
        kxf, kyf, kzf =  np.array([kx, ky, kz]) @ np.linalg.inv(B)
        # print(kx, ky, kz)
        # break

        energies    = np.array(find(ks, "eigenvalues").text.split(), dtype=float) * Energy_Unit_Conv
        occupations = np.array(find(ks, "occupations").text.split(), dtype=float)

        # optional: spin attribute
        spin = int(ks.attrib.get("spin", 1))

        for ib in range(nbnd):
            records.append({
                "ik":     ik,
                "ib":     ib,
                "spin":   spin,
                "kx":     kxf,
                "ky":     kyf,
                "kz":     kzf,
                "kx_cart":     kx,
                "ky_cart":     ky,
                "kz_cart":     kz,
                "weight": weight,
                "E_Ry":   energies[ib],
                "E_Ry_relative": energies[ib] - ef,
                "E_eV_relative": (energies[ib] - ef) * Ry_to_eV,
                "occ":    occupations[ib],
            })
            pass
        pass


    inp = find(root, ".//input")
    # atomic_structure attributes
    atomic_structure = find(inp, "atomic_structure")

    nat = int(atomic_structure.get("nat"))
    alat = float(atomic_structure.get("alat"))
    bravais_index = int(atomic_structure.get("bravais_index"))

    print(nat, alat, bravais_index)



    species_block = find(inp, "atomic_species")

    species = []
    for sp in findall(species_block, "species"):
        species.append({
            "name": sp.get("name"),
            "mass": float(find(sp, "mass").text),
            "pseudo_file": find(sp, "pseudo_file").text
        })

    print(species)

    positions = find(atomic_structure, "atomic_positions")
    # atoms = []
    # for atom in findall(positions, "atom"):
    #     coords = list(map(float, atom.text.split()))

    #     atoms.append({
    #         "name": atom.get("name"),
    #         "index": int(atom.get("index")),
    #         "coords": coords
    #     })

    # print(atoms)

    atoms = []
    for atom in findall(positions, "atom"):
        coords = atom.text
        atoms.append("{}  {}".format(atom.get("name"), coords))
        pass

    print(atoms)



    cell = find(atomic_structure, "cell")

    # cell_vectors = [
    #     "a1": list(map(float, find(cell, "a1").text.split())),
    #     "a2": list(map(float, find(cell, "a2").text.split())),
    #     "a3": list(map(float, find(cell, "a3").text.split()))
    # ]

    cell_vectors = [find(cell, "a1").text, find(cell, "a2").text, find(cell, "a3").text]
    print(cell_vectors)


    df = pd.DataFrame(records)
    return df, ef, nbnd, nks, cell_vectors, atoms


def write_boltztrap_structure(filepath, cell_vectors, atoms):
    with open(filepath, "w") as f:
        f.write("BoltzTraP geometry file, generated from QE data-file-schema.xml with custome code \n")
        f.write("\n".join(cell_vectors) + "\n")
        f.write("{}\n".format(len(atoms)))
        f.write("\n".join(atoms) + "\n")

        pass

    pass



def write_boltztrap_energy(df, filepath, EFermi):
    """
    Write BoltzTraP2-compatible energy file from parsed QE dataframe.
    
    Format per k-point:
        kx ky kz Nb
        E_1
        E_2
        ...
        E_Nb
    """
    prefix = "BoltzTraP eigen-energies file, generated from QE data-file-schema.xml with custome code in given energy range"
    
    # group by k-point index, preserving k-coords
    grouped = df.sort_values(["ik", "ib"]).groupby("ik")

    header = "{}\t{}\t{} ! nk, nspin, Fermi level(Ry) : energies below in Ry".format(len(grouped), 1, EFermi)

    lines = []
    lines.append(f"{prefix}")                      # header line: system name
    lines.append(f"{header}")                      # header line: system name
    # lines.append(f"{len(grouped)}")                # total number of k-points. header contains the group information

    
    for ik, group in grouped:
        group = group.sort_values("ib")
        kx = group["kx"].iloc[0]
        ky = group["ky"].iloc[0]
        kz = group["kz"].iloc[0]
        nb = len(group)

        # BoltzTraP2 expects energies in Ry?
        energies_Ry = group["E_Ry"].values

        lines.append(f"  {kx:.8f}  {ky:.8f}  {kz:.8f}  {nb} ! kpt nband")
        for e in energies_Ry:
            lines.append(f"  {e:.8f}")

    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Written {len(grouped)} k-points, {nb} bands → {filepath}")


def write_energy_blocks(df, filepath):
    """
    Minimal block format:
        kx ky kz Nb
        E_1
        ...
        E_Nb
    (no header, energies in Ry as-is from dataframe)
    """
    grouped = df.sort_values(["ik", "ib"]).groupby("ik")

    with open(filepath, "w") as f:
        for ik, group in grouped:
            group = group.sort_values("ib")
            kx, ky, kz = group[["kx","ky","kz"]].iloc[0]
            nb = len(group)
            f.write(f"{kx:.8f}  {ky:.8f}  {kz:.8f}  {nb}\n")
            for e in group["E_Ry"].values:
                f.write(f"  {e:.10f}\n")

    print(f"Done: {filepath}")
    




def compute_energy_files(qe_xml_file, out_dir, prefix, erange=()):
    """
    Compute .energy files for botlztrap2
    
    """

    df, ef, nbnd, nks, cell_vectors, atoms = parse_qe_xml(qe_xml_file)
    print(df.head(10))
    print(df.dtypes)

    write_boltztrap_structure(out_dir + "/{}.structure".format(prefix), cell_vectors, atoms)
    


    df2 = df[df["E_eV_relative"] < erange[1]]
    df2 = df2[df2["E_eV_relative"] > erange[0]]

    ib_idx = df2['ib'].unique()

    df3 = df[df['ib'].isin(ib_idx)]

    write_boltztrap_energy(df3, out_dir + "/{}.energy".format(prefix), ef)

    for ib in ib_idx:
        df3 = df[df['ib']== ib]
        from pathlib import Path
        dir_name = out_dir + "/ib{}".format(ib)
        Path(dir_name).mkdir(exist_ok=True)
        print(dir_name)
        write_boltztrap_energy(df3, dir_name + "/{}_ib{}.energy".format(prefix, ib), ef)
        write_boltztrap_structure(dir_name + "/{}_ib{}.structure".format(prefix, ib), cell_vectors, atoms)
        pass




def plot_k_path(data_dir, btp_file, kpath=None, niter=1, unit_conv=1):
    """

    unit_conv : default 1 . If you want in eV then 
    """

    data, equivalences, coeffs = load_interpolation(data_dir, btp_file, niter=niter)


    if kpath is None:
        # G-M-K-G-A-L-H-A-L-M-K-H
        kpath_list="[0.0,0.0,0.0], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], " \
        "[0.0,0.0,0.0], [0.0,0.0,0.5], [0.5,0.0,0.5], [0.333333,0.333333,0.5], " \
        "[0.0,0.0,0.5], [0.5,0.0,0.5], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], [0.333333,0.333333,0.5]"
        nkpoints_list=np.array([39, 23, 45, 94, 29, 23, 45, 1, 94, 1, 94, 1])*5

    # The second position alargument is first interpreted as a Python literal,
    # and after parsing it is cast to a NumPy array, which must have the right
    # dimensions. The special value None directs the parser to split the path
    # in several parts.
    try:
        kpaths = ast.literal_eval(kpath_list)
    except ValueError:
        exit("cannot be parsed as a Python literal")
    if not isinstance(kpaths, list):
        exit("cannot be parsed as a Python list")
    kpaths = [
        list(group)
        for k, group in itertools.groupby(kpaths, key=lambda x: x is not None)
        if k
    ]

    nkpoints=100

    kpaths = [np.array(i, dtype=np.float64) for i in kpaths]        

    plt.figure()
    ax = plt.gca()
    ticks = []
    dividers = []
    offset = 0.0
    for ikpath, kpath in enumerate(kpaths):
        ax.set_prop_cycle(
            color=matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        )
        info("k path #{}".format(ikpath + 1))
        # Generate the explicit point list.
        band_path = asekp.bandpath(kpath, data.atoms.cell, nkpoints_list[ikpath])
        if isinstance(band_path, asekp.BandPath):
            # For newer versions of ASE.
            kp = band_path.kpts
            dkp, dcl = band_path.get_linear_kpoint_axis()[:2]
        else:
            # For older versions of ASE.
            kp, dkp, dcl = band_path
        dkp += offset
        dcl += offset
        # Compute the band energies
        with TimerContext() as timer:
            egrid = fite.getBands(
                kp, equivalences, data.get_lattvec(), coeffs
            )[0]
            deltat = timer.get_deltat()
            info("rebuilding the bands took {:.3g} s".format(deltat))
        egrid -= data.fermi
        # Create the plot
        nbands = egrid.shape[0]
        for i in range(nbands):
            plt.plot(dkp, egrid[i, :]*unit_conv, lw=2.0)
        ticks += dcl.tolist()
        dividers += [dcl[0], dcl[-1]]
        offset = dkp[-1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([])
    for d in ticks:
        plt.axvline(x=d, ls="--", lw=0.5)
    for d in dividers:
        plt.axvline(x=d, ls="-", lw=2.0)
    plt.axhline(y=0.0, lw=1.0)
    plt.ylabel(r"$\varepsilon - \varepsilon_F\;\left[\mathrm{Ha}\right]$")
    plt.ylim(-.05,.05)
    if unit_conv != 1:
        plt.ylabel(r"$\varepsilon - \varepsilon_F\;\left[\mathrm{Ha}\right]$" + "x{:.2f}".format(unit_conv))
        plt.ylim((-.05*unit_conv,.05*unit_conv))
        pass

    plt.tight_layout()
    

    plt.savefig("test.png")
    pass



def plot_dos_TDF(data_dir, btp_file, kpath=None, niter=1):
    n_bins = 1000
    data, equivalences, coeffs = load_interpolation(data_dir, btp_file, niter=niter)
    lattvec = data.get_lattvec()
    eband, vvband, cband = fite.getBTPbands(
        equivalences, coeffs, lattvec, curvature=False
    )

    Cepsilon, Cdos, Cvvdos, cdos = BL.BTPDOS(eband, vvband, npts=n_bins)


    plt.plot((Cepsilon - data.fermi)/units.eV, Cdos*units.eV)
    pass




if __name__ == "__main__":
    # compute_energy_files("./data-file-schema.xml", out_dir="./energy", prefix="test", erange=(-0.1,0.1))
    plot_k_path(".", "./test.bt2", niter=2, unit_conv=Ha_to_eV)


    pass




