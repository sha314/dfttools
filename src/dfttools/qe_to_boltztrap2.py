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


from dfttools import compute, compute_TEP, smooth


Ha_to_eV = 27.211396132
Ry_to_eV = 13.6057039763
Ry_to_Ha = Ry_to_eV/Ha_to_eV
Ha_to_Ry = 1.0/Ry_to_Ha




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
    ef   = float(find(band_structure, "fermi_energy").text.strip()) * Ha_to_Ry


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

        energies    = np.array(find(ks, "eigenvalues").text.split(), dtype=float) * Ha_to_Ry
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
    

def parse_data_file_schema(xml_path: str) -> Dict:
    """
    Parse Quantum ESPRESSO's data-file-schema.xml to extract:
    - Lattice vectors (A=at), reciprocal lattice vectors (bg), alat is the cell parameter scale

    Reciprocal matrix, B = 2*pi * inverse(at) -> used in FermiSurfer
    Also, B == 2 pi/alat * transpose(bg)

    B @ A = 2 pi is satisfied

    the bg vector from xml file. Each row is a reciprocal lattice vector b * alat/(2 pi). That is
            b1
    bg =    b2
            b3

    To get B, you need 2 pi/alat * transpose(bg).

    Now,
             
    K_cart = b1 k1 + b2 k2 + b3 k3 =
    K_cart =  B . K_frac

    both K_cart and K_frac are collumn vectors and B is a matrix with collumns [b1,b2,b3] lattice vectors.


    - k-points (xk), weights, nk1, nk2, nk3 (Monkhorst-Pack grid)
    - k1, k2, k3 (grid offset)
    - Band energies (et), nbnd, nks
    - Fermi energy (ef), ef_up, ef_dw
    - nspin, two_fermi_energies
    - Symmetry operations (s, nsym, time_reversal, t_rev)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data = {}

    # --- Cell parameters ---
    cell = root.find('.//cell')
    # Direct lattice vectors (at) in alat units: columns are vectors
    a1 = cell.find('a1')
    a2 = cell.find('a2')
    a3 = cell.find('a3')
    at = np.array([
        [float(x) for x in a1.text.split()],
        [float(x) for x in a2.text.split()],
        [float(x) for x in a3.text.split()]
    ]).T  # Fortran convention: at(:,i) is the i-th vector, so transpose

    # Reciprocal lattice vectors (bg) in 2π/alat units
    reciprocal_lattice = root.find(".//reciprocal_lattice")
    b1 = reciprocal_lattice.find('b1')
    b2 = reciprocal_lattice.find('b2')
    b3 = reciprocal_lattice.find('b3')
    # print("b1 ", b1)
    bg = np.array([
        [float(x) for x in b1.text.split()],
        [float(x) for x in b2.text.split()],
        [float(x) for x in b3.text.split()]
    ]).T

    alat = float(root.find('.//atomic_structure').attrib.get('alat', 1.0))
    data['alat'] = alat
    data['at'] = at  # 3x3 matrix, columns are lattice vectors in alat units
    data['bg'] = bg  # 3x3 matrix, columns are reciprocal vectors in 2π/alat units


    # print(f"Raw XML: alat={alat:.4f} Bohr")
    # print(f"  at columns (first): {at[:,0]}")
    # print(f"  bg columns (first): {bg[:,0]}")

    # # Normalize at and bg to match Fortran conventions
    # at, bg, alat = normalize_at_bg(at, bg, alat)

    # print(f"  at columns (first): {at[:,0]}")
    # print(f"  bg columns (first): {bg[:,0]}")

    # data['alat'] = alat
    # data['at'] = at
    # data['bg'] = bg

    # print(f"  After normalization: at_zz={at[2,2]:.4f}, bg_zz={bg[2,2]:.4f}")
    # print(f"  Check at^T @ bg = I: {np.allclose(np.dot(at.T, bg), np.eye(3), atol=1e-3)}")





    # --- Monkhorst-Pack grid ---
    # Look for monkhorst_pack or k_points_IBZ
    mp_grid = root.find('.//monkhorst_pack')
    if mp_grid is not None:
        data['nk1'] = int(mp_grid.attrib['nk1'])
        data['nk2'] = int(mp_grid.attrib['nk2'])
        data['nk3'] = int(mp_grid.attrib['nk3'])

        # Center of computation, Default Gamma=0,0,0
        data['k1'] = int(mp_grid.attrib['k1'])
        data['k2'] = int(mp_grid.attrib['k2'])
        data['k3'] = int(mp_grid.attrib['k3'])
    else:
        # Try to infer from k_points_list
        k_points = root.findall('.//k_point')
        data['nk1'] = data['nk2'] = data['nk3'] = len(k_points)
        data['k1'] = data['k2'] = data['k3'] = 0

    # --- Band energies ---
    band_structure = root.find(".//band_structure")
    nbnd = int(band_structure.find(".//nbnd").text)
    nelec = float(band_structure.find(".//nelec").text)
    ef = fermi_energy = float(band_structure.find(".//fermi_energy").text)
    data['ef'] = ef
    data['fermi_energy'] = ef
    data['ef_unit'] = "Ha"
    nks = int(band_structure.find(".//nks").text)


    xk = []
    et = []
    xkf = [] # in fractional coordinate
    for ik, ks in enumerate(band_structure.findall("ks_energies")):
        kpt = ks.find("k_point")
        kx, ky, kz = [float(x) for x in kpt.text.split()]
        xk.append([kx, ky, kz])
        energies  = np.array(ks.find("eigenvalues").text.split(), dtype=float)
        et.append(energies)
        pass
   
    data['et'] = np.array(et)  # (nbnd, nks) in Ry
    data['nbnd'] = nbnd
    data['nks'] = nks
    data['xk'] = np.array(xk)  # (3, nks) in crystal coordinates (2π/alat units) or dimension less coordiante

    # --- Spin info ---
    # Check for nspin in the XML
    nspin_elem = root.find('.//nspin')
    data['nspin'] = int(nspin_elem.text) if nspin_elem is not None else 1

    # --- Fermi energy ---
    # Try different paths
    # ef_elem = root.find('.//fermi_energy')
    # if ef_elem is not None:
    #     data['ef'] = float(ef_elem.text)
    # else:
    #     data['ef'] = 0.0

    # ef_up_elem = root.find('.//two_fermi_energies/fermi_energy_up')
    # ef_dw_elem = root.find('.//two_fermi_energies/fermi_energy_down')
    # if ef_up_elem is not None and ef_dw_elem is not None:
    #     data['ef_up'] = float(ef_up_elem.text)
    #     data['ef_dw'] = float(ef_dw_elem.text)
    #     data['two_fermi_energies'] = True
    # else:
    #     data['ef_up'] = data['ef']
    #     data['ef_dw'] = data['ef']
    #     data['two_fermi_energies'] = False

    # --- Symmetries ---
    symmetries = root.find('.//symmetries')
    symmetry_tmp = symmetries.findall("symmetry")
    nsym = len(symmetry_tmp)
    symm_mat = np.zeros((3, 3, nsym), dtype=int)
    trans_mat = np.zeros((3, nsym), dtype=int)
    t_rev = np.zeros(nsym, dtype=int)
    for isym, sym in enumerate(symmetry_tmp):
        rotation = sym.find("rotation")
        translation = sym.find("fractional_translation")

        if rotation is not None:
            # print("rot.text ", rot.text)
            rot_mat = np.array([float(x) for x in rotation.text.split()]).reshape(3, 3, order="C")
            # print(rot_mat)
            symm_mat[:, :, isym] = rot_mat
            pass

        if translation is not None:
            t_mat = np.array([float(x) for x in translation.text.split()])
            trans_mat[:, isym] = t_mat
            pass



    # for isym, sym in enumerate(symmetries):
    #     # Rotation matrix (in crystal coordinates, integer entries)
    #     rot = sym.find('rotation')
    #     if rot is not None:
    #         # print("rot.text ", rot.text)
    #         rot_mat = np.array([float(x) for x in rot.text.split()]).reshape(3, 3, order="C")
    #         print(rot_mat)
    #         symm_mat[:, :, isym] = rot_mat

    #     # Time reversal
    #     trev = sym.find('time_reversal')
    #     if trev is not None:
    #         t_rev[isym] = int(trev.text)

    data['nsym'] = nsym
    data['rotation'] = symm_mat
    data['translation'] = trans_mat
    # data['t_rev'] = t_rev

    # Check if time_reversal is used at all
    # data['time_reversal'] = any(t_rev == 1) or any(t_rev == -1)

    
    inp = root.find(".//input")
    # atomic_structure attributes
    atomic_structure = inp.find("atomic_structure")

    nat = int(atomic_structure.get("nat"))
    alat = float(atomic_structure.get("alat"))
    bravais_index = int(atomic_structure.get("bravais_index"))

    data["nat"] = nat
    data["alat"] = alat
    data["bravais_index"] = bravais_index
    

    species_block = inp.find("atomic_species")

    species = []
    for sp in species_block.findall("species"):
        species.append({
            "name": sp.get("name"),
            "mass": float(sp.find("mass").text),
            "pseudo_file": sp.find("pseudo_file").text
        })


    positions = atomic_structure.find("atomic_positions")
    

    atoms = []
    for atom in positions.findall("atom"):
        coords = atom.text
        atoms.append("{}  {}".format(atom.get("name"), coords))
        pass


    data["atomic_species"] = species
    data["atomic_positions"] = atoms

    return data

    
def get_data_frame(data_dict):
    """
    energy in Hartee unit
    """
    records = []
    BINV = np.linalg.inv(data_dict['bg'])
    print(BINV)
    spin = data_dict['nspin']
    ef = data_dict['ef']
    for ik, kpt in enumerate(data_dict['xk']):
        kx, ky, kz = kpt
        kxf, kyf, kzf = kpt @ BINV

        energies = data_dict['et'][ik]
        for ib, eVal in enumerate(energies):
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
                    "E_Ha":   eVal,
                    "E_Ha_relative": eVal - ef,
                    "E_eV_relative": (eVal - ef) * Ha_to_eV
                })
        pass
    df = pd.DataFrame(records)
    return df





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


def compute_energy_files_v2(qe_xml_file, out_dir, prefix, erange=(-0.2,0.2)):
    """
    Compute .energy files for botlztrap2
    
    """

    data = parse_data_file_schema(xml_file)
    df = get_data_frame(data)

    df['E_Ry'] = df["E_Ha"]*Ha_to_Ry
    ef = data['ef']
    nbnd = data['nbnd']
    nks = data['nks']
    atoms  = data['atomic_positions']

    cell_vectors = [f"{a1} {a2} {a3}" for a1,a2,a3 in data['at']]


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


def get_bands_by_energy_from_interpolation(btp_file, out_dir, prefix, erange=()):
    """
    get bands in given energy range

    erange:
    
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

def find_bands_in_e_range(data, equivalences, coeffs, erange):

    pass




def plot_k_path_for_n_bands_ax(data, equivalences, coeffs, band_ids=None, kpath=None, unit_conv=1, axes=None, colors=None, shift_eV=0):
    """
    data, equivalences, coeffs : returned by interpolation method load_interpolation. Takes matplotlib axes as argument and plots there

    band_ids : band ids of interest
    kpath : a string of lists of k point. Default '[0.0,0.0,0.0], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], " \
        "[0.0,0.0,0.0], [0.0,0.0,0.5], [0.5,0.0,0.5], [0.333333,0.333333,0.5], " \
        "[0.0,0.0,0.5], [0.5,0.0,0.5], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], [0.333333,0.333333,0.5]' for patj

    """
    coeffs_tmp = coeffs
    if band_ids is not None:
        # data_tmp.ebands = data.ebands[band_ids,:] # actual data is not needed because boltztrap interpolates using equivalences and coeffs
        coeffs_tmp = coeffs[band_ids,:]
        pass
    nkpoints=5
    if kpath is None:
        # G-M-K-G-A-L-H-A-L-M-K-H
        kpath_list="[0.0,0.0,0.0], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], " \
        "[0.0,0.0,0.0], [0.0,0.0,0.5], [0.5,0.0,0.5], [0.333333,0.333333,0.5], " \
        "[0.0,0.0,0.5], [0.5,0.0,0.5], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], [0.333333,0.333333,0.5]"
        nkpoints_list=np.array([39, 23, 45, 94, 29, 23, 45, 1, 94, 1, 94, 1])*nkpoints
    else:
        kpath_list = kpath[0]
        nkpoints_list = kpath[1]
        pass
    

    # The second position alargument is first interpreted as a Python literal,
    # and after parsing it is cast to a NumPy array, which must have the right
    # dimensions. The special value None directs the parser to split the path
    # in several parts.
    try:
        kpaths = ast.literal_eval(kpath_list)
    except ValueError:
        print("cannot be parsed as a Python literal")
    if not isinstance(kpaths, list):
        print("cannot be parsed as a Python list")
    kpaths = [
        list(group)
        for k, group in itertools.groupby(kpaths, key=lambda x: x is not None)
        if k
    ]

    

    kpaths = [np.array(i, dtype=np.float64) for i in kpaths]        

    if axes is None:
        print("axes is None ")
        return None
        pass
    ax_ = axes
    ticks = []
    dividers = []
    offset = 0.0
    for ikpath, kpath in enumerate(kpaths):
        ax_.set_prop_cycle(
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
                kp, equivalences, data.get_lattvec(), coeffs_tmp
            )[0]
            deltat = timer.get_deltat()
            info("rebuilding the bands took {:.3g} s".format(deltat))
        egrid -= (data.fermi + shift_eV/Ha_to_eV)
        # Create the plot
        nbands = egrid.shape[0]
        print(nbands)
        for i in range(nbands):
            if colors is None:
                ax_.plot(dkp, egrid[i, :]*unit_conv, lw=2.0)
            else:
                ax_.plot(dkp, egrid[i, :]*unit_conv, lw=2.0, color=colors[i])
        ticks += dcl.tolist()
        dividers += [dcl[0], dcl[-1]]
        offset = dkp[-1]
    ax_.set_xticks(ticks)
    ax_.set_xticklabels([])
    for d in ticks:
        ax_.axvline(x=d, ls="--", lw=0.5)
    for d in dividers:
        ax_.axvline(x=d, ls="-", lw=2.0)
    ax_.axhline(y=0.0, lw=1.0)
    ax_.set_ylabel(r"$\varepsilon - \varepsilon_F\;\left[\mathrm{Ha}\right]$")
    ax_.set_ylim(-.05,.05)
    if unit_conv != 1:
        ax_.set_ylabel(r"$\varepsilon - \varepsilon_F\;\left[\mathrm{Ha}\right]$" + "x{:.2f}".format(unit_conv))
        ax_.set_ylim((-.05*unit_conv,.05*unit_conv))
        pass
    
    return ax_


def plot_k_path_for_n_bands(data, equivalences, coeffs, band_ids=None, kpath=None, unit_conv=1):
    """
    data, equivalences, coeffs : returned by interpolation method load_interpolation

    band_ids : band ids of interest
    kpath : a string of lists of k point. Default '[0.0,0.0,0.0], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], " \
        "[0.0,0.0,0.0], [0.0,0.0,0.5], [0.5,0.0,0.5], [0.333333,0.333333,0.5], " \
        "[0.0,0.0,0.5], [0.5,0.0,0.5], [0.5, 0.0, 0.0], [0.333333,0.333333,0.0], [0.333333,0.333333,0.5]' for patj

    """
    plt.figure()
    ax = plt.gca()
    plot_k_path_for_n_bands_ax(data, equivalences, coeffs, band_ids=band_ids, kpath=kpath, unit_conv=unit_conv, axes=ax)

    plt.tight_layout()
    plt.savefig("test.png")
    



def plot_k_path(data_dir, btp_file, niter=1, band_ids=None, kpath=None, unit_conv=1, axes=None):
    """

    unit_conv : default 1 . If you want in eV then 
    """

    data, equivalences, coeffs = compute.load_interpolation(data_dir, btp_file, niter=niter)

    return plot_k_path_for_n_bands_ax(data, equivalences, coeffs, band_ids, kpath, unit_conv, axes)
    




def plot_dos_TDF(data_dir, btp_file, kpath=None, niter=1):
    n_bins = 1000
    data, equivalences, coeffs = compute.load_interpolation(data_dir, btp_file, niter=niter)
    lattvec = data.get_lattvec()
    eband, vvband, cband = fite.getBTPbands(
        equivalences, coeffs, lattvec, curvature=False
    )

    Cepsilon, Cdos, Cvvdos, cdos = BL.BTPDOS(eband, vvband, npts=n_bins)


    plt.plot((Cepsilon - data.fermi)/units.eV, Cdos*units.eV)
    pass





def compute_Szz_Sxx_v2(dft_data_dir, shift, ylimList, fig_name):


    filename_DOS = dft_data_dir      + "Nb3S4_PBEsol_NCPP_QE_data_DOS.csv"
    filename_TDF_CRTA = dft_data_dir +  "Nb3S4_PBEsol_NCPP_QE_data_TDF_CRTA.csv"
    filename_TDF_IDOS = dft_data_dir +  "Nb3S4_PBEsol_NCPP_QE_data_TDF_IDOS.csv"

    energy, DOS = np.loadtxt(filename_DOS, delimiter=',').T

    energy1, TDF_xx_CRTA, TDF_yy_CRTA, TDF_zz_CRTA = np.loadtxt(filename_TDF_CRTA, delimiter=',').T

    energy2, TDF_xx_IDOS, TDF_yy_IDOS, TDF_zz_IDOS = np.loadtxt(filename_TDF_IDOS, delimiter=',').T

    Ep = shift/1000.
    x, idx = compute_TEP.shift_and_clip(energy, Ep, erange=(-0.6, 0.6))
    TDF_zz = TDF_zz_CRTA[idx]
    TDF_xx = TDF_xx_CRTA[idx]
    DOS_values = DOS[idx]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=200)
    axes = axes.flatten()


    ax = axes[0]
    ax.plot(energy, DOS, label="DOS")
    ax.plot(x, DOS_values, label="DOS, shifted")
    ax.set_xlabel("(E-E_F-{}meV ) (eV)".format(Ep*1000))
    ax.axvline(0, 0, 10, color='red')
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim(0, 8)
    ax.legend()


    ax = axes[1]
    ax.plot(x, TDF_zz_IDOS[idx], label="TDF_zz_IDOS")
    ax.plot(x, TDF_zz_CRTA[idx], label="TDF_zz_CRTA")

    ax.plot(energy, -compute_TEP.dFD_dE(energy, 0, 300)/30, 'k--', label="-df/dE(T=300K)/30")
    ax.axvline(0, 0, 10, color='red')
    ax.set_xlabel("(E-E_F-{}meV ) (eV)".format(Ep*1000))
    ax.set_ylabel("Shifted TDF")
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim(0, 1.5)
    ax.legend()



    ax = axes[2]
    ax.plot(x, TDF_xx_IDOS[idx], label="TDF_xx_IDOS")
    ax.plot(x, TDF_xx_CRTA[idx], label="TDF_xx_CRTA")

    ax.plot(energy, -compute_TEP.dFD_dE(energy, 0, 300)/50, 'k--', label="-df/dE(T=300K)/50")
    ax.axvline(0, 0, 10, color='red')
    ax.set_xlabel("(E-E_F-{}meV ) (eV)".format(Ep*1000))
    ax.set_ylabel("Shifted TDF")
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim(0, 0.3)
    ax.legend()

    # plot_k_path(dft_data_dir, dft_data_dir + "/test.bt2", None, axes=axes[3])
    data, equivalences, coeffs = compute.load_interpolation(dft_data_dir, dft_data_dir + "test.bt2", niter=1)
    plot_k_path_for_n_bands_ax(data, equivalences, coeffs, band_ids=None, kpath=None, unit_conv=Ha_to_eV, axes=axes[3])

    # plot_k_path(dft_data_dir, dft_data_dir + "test.bt2", niter=1, band_ids=None, kpath=None, unit_conv=Ha_to_eV, axes=axes[3])
    

    ax = axes[4]
    # ax.plot(Tzzdata, Szzdata, 'o', label="Experiment", alpha=0.6)



    
    Tlist = np.linspace(10, 400, 20)
    SzzIDOS = compute_TEP.get_thermopower(x, TDF_zz_IDOS[idx], Tlist)
    SzzCRTA = compute_TEP.get_thermopower(x, TDF_zz_CRTA[idx], Tlist)
    ax.plot(Tlist, SzzIDOS*10**6, 'r--', label=r"$\mu(T)=0$ 1/DOS")
    ax.plot(Tlist, SzzCRTA*10**6, 'g--', label=r"$\mu(T)=0$ CRTA")


    # SzzIDOS = get_thermopower(x, TDF_zz_IDOS[idx], Tlist, mulist=muT(Tlist))
    # SzzCRTA = get_thermopower(x, TDF_zz_CRTA[idx], Tlist, mulist=muT(Tlist))
    # ax.plot(Tlist, SzzIDOS*10**6, 'r-', label=r"$\mu(T)$ 1/DOS")
    # ax.plot(Tlist, SzzCRTA*10**6, 'g-', label=r"$\mu(T)$ CRTA")



    ax.set_ylim(-60, 20)
    ax.set_xlabel(r"Temperature (K)")
    ax.set_ylabel(r"Thermopwoer, $S_{zz}\, \mu V/K$")
    ax.legend()



    ax = axes[5]
    # ax.plot(T1xxdata, S1xxdata, 'o', label="Nb3S4-002-13", alpha=0.6)
    # ax.plot(T2xxdata, S2xxdata, 'x', label="Nb3S4-3", alpha=0.6)

        
    Tlist = np.linspace(10, 400, 20)
    SxxIDOS = compute_TEP.get_thermopower(x, TDF_xx_IDOS[idx], Tlist)
    SxxCRTA = compute_TEP.get_thermopower(x, TDF_xx_CRTA[idx], Tlist)
    ax.plot(Tlist, SxxIDOS*10**6, 'r--', label=r"$\mu(T)=0$ 1/DOS")
    ax.plot(Tlist, SxxCRTA*10**6, 'g--', label=r"$\mu(T)=0$ CRTA")


    # SxxIDOS = get_thermopower(x, TDF_xx_IDOS[idx], Tlist, mulist=muT(Tlist))
    # SxxCRTA = get_thermopower(x, TDF_xx_CRTA[idx], Tlist, mulist=muT(Tlist))
    # ax.plot(Tlist, SxxIDOS*10**6, 'r-', label=r"$\mu(T)$ 1/DOS")
    # ax.plot(Tlist, SxxCRTA*10**6, 'g-', label=r"$\mu(T)$ CRTA")



    # ax.set_ylim(-10, 120)

    for i, ax in enumerate(axes):
        ax.set_ylim(ylimList[i])
        pass

    ax.set_xlabel(r"Temperature (K)")
    ax.set_ylabel(r"Thermopwoer, $S_{xx}\, \mu V/K$")
    ax.legend()
    return fig, axes
    


# Make data available globally
T1xxdata, S1xxdata, T2xxdata, S2xxdata = np.loadtxt("Nb3S4-a-axis-thermopower.csv", delimiter=',', usecols=(0,1,2,3)).T
Tzzdata, Szzdata = np.loadtxt("Nb3S4-c-axis-thermopower.txt").T


def band_resolved_S_CRTA(shift, constC):
    """
    Requrements: ROOT_DIR contains .energy and .structure files for 4 bands and sub directories contains .energy and .structure files
    for individual bands.

    shift  : in meV of the fermi level
    constC : contribution factor of electron like bands relative to hole like bands

    
    """
    ROOT_DIR = "./PBEsol-Relaxed/energy/"

    dir_dict = {
        "4bands" : ROOT_DIR,
        "ib61" : ROOT_DIR + "ib61/",
        "ib62" : ROOT_DIR + "ib62/",
        "ib63" : ROOT_DIR + "ib63/",
        "ib64" : ROOT_DIR + "ib64/"
    }

    color_for_bands = ["red", "green", "blue", "orange"]
    colors = {
        "4bands" : "k",
        "ib61" : color_for_bands[0],
        "ib62" : color_for_bands[1],
        "ib63" : color_for_bands[2],
        "ib64" : color_for_bands[3]
    }


    sigma_param = 3
    sigma_dict = {"segments": [0.08, 0.13], "sigma": [2, 4, 6]}

    # sigma_dict = {"segments": [0.08], "sigma": [2, 5]}
    Ep = shift/1000.
    # Ep = 0.


    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=200)
    axes = axes.flatten()


    # plot_k_path(dft_data_dir, dft_data_dir + "/test.bt2", None, axes=axes[3])
    data, equivalences, coeffs = compute.load_interpolation(ROOT_DIR, ROOT_DIR + "test.bt2", niter=1)
    plot_k_path_for_n_bands_ax(data, equivalences, coeffs, band_ids=None, kpath=None, 
                                                unit_conv=Ha_to_eV, axes=axes[3], colors=color_for_bands, shift_eV=Ep)

    # plot_k_path(dft_data_dir, dft_data_dir + "test.bt2", niter=1, band_ids=None, kpath=None, unit_conv=Ha_to_eV, axes=axes[3])


    axes[4].plot(Tzzdata, Szzdata, 'o', label="Experiment", alpha=0.6)
    axes[5].plot(T1xxdata, S1xxdata, 'o', label="Nb3S4-002-13", alpha=0.6)
    axes[5].plot(T2xxdata, S2xxdata, 'x', label="Nb3S4-3", alpha=0.6)

    Tlist = np.linspace(10, 400, 20)

    Sxx_sum = None
    Szz_sum = None

    onsgar_dict = dict()

    for key in dir_dict:
        dft_data_dir = dir_dict[key]

        filename_DOS = dft_data_dir      + "Nb3S4_PBEsol_NCPP_QE_data_DOS.csv"
        filename_TDF_CRTA = dft_data_dir +  "Nb3S4_PBEsol_NCPP_QE_data_TDF_CRTA.csv"
        filename_TDF_IDOS = dft_data_dir +  "Nb3S4_PBEsol_NCPP_QE_data_TDF_IDOS.csv"

        energy, DOS = np.loadtxt(filename_DOS, delimiter=',').T

        energy1, TDF_xx_CRTA, TDF_yy_CRTA, TDF_zz_CRTA = np.loadtxt(filename_TDF_CRTA, delimiter=',').T

        energy2, TDF_xx_IDOS, TDF_yy_IDOS, TDF_zz_IDOS = np.loadtxt(filename_TDF_IDOS, delimiter=',').T

        
        x, idx = compute_TEP.shift_and_clip(energy, Ep, erange=(-0.6, 0.6))
        TDF_zz = TDF_zz_CRTA[idx]
        TDF_xx = TDF_xx_CRTA[idx]
        DOS_values = DOS[idx]


        ax = axes[0]
        # ax.plot(energy, smooth.gaussian_smooth_1d_reflected(DOS, sigma_param), label="{}".format(key))
        # ax.plot(x, smooth.gaussian_smooth_1d_reflected(DOS_values, sigma_param), label="{}".format(key), color=colors[key])
        ax.plot(x, smooth.segmented_gaussian_smooth_v2(x, DOS_values, sigma_dict)[0], label="{}".format(key), color=colors[key])


        sigma_param = 6
        # sigma_dict = {"segments": [0.01,], "sigma": [4, 8]}
        ax = axes[1]
        # ax.plot(x, smooth.gaussian_smooth_1d_reflected(TDF_zz_IDOS[idx], sigma_param), label="TDF_zz_IDOS")
        ax.plot(x, smooth.gaussian_smooth_1d_reflected(TDF_zz_CRTA[idx], sigma_param), label="{}".format(key), color=colors[key])
        # ax.plot(x, smooth.segmented_gaussian_smooth_v2(x, TDF_zz_CRTA[idx], sigma_dict)[0], label="{}".format(key), color=colors[key])

        sigma_param = 4
        ax = axes[2]
        # ax.plot(x, smooth.gaussian_smooth_1d_reflected(TDF_xx_IDOS[idx], sigma_param), label="TDF_xx_IDOS")
        ax.plot(x, smooth.gaussian_smooth_1d_reflected(TDF_xx_CRTA[idx], sigma_param), label="{}".format(key), color=colors[key])
        # ax.plot(x, smooth.segmented_gaussian_smooth_v2(x, TDF_xx_CRTA[idx], sigma_dict)[0], label="{}".format(key), color=colors[key])


        

        ax = axes[4]
        onsgar_zz = compute_TEP.get_OnsagerCoeff(x, TDF_zz_CRTA[idx], Tlist)
        SzzCRTA = onsgar_zz[:,1]/onsgar_zz[:,0]/(-Tlist)
        ax.plot(Tlist, SzzCRTA*10**6, label="{}".format(key), color=colors[key])



        ax = axes[5]  
        onsgar_xx = compute_TEP.get_OnsagerCoeff(x, TDF_xx_CRTA[idx], Tlist)
        SxxCRTA = onsgar_xx[:,1]/onsgar_xx[:,0]/(-Tlist)
        ax.plot(Tlist, SxxCRTA*10**6,  label="{}".format(key), color=colors[key])

        if key != "4bands":
            onsgar_dict[key] = {'zz':onsgar_zz, 'xx':onsgar_xx}
            pass


        if Sxx_sum is None:
            Sxx_sum = SxxCRTA
        else:
            Sxx_sum += SxxCRTA

        if Szz_sum is None:
            Szz_sum = SzzCRTA
        else:
            Szz_sum += SzzCRTA

        # ax.set_ylim(-10, 120)



    pass


    ax = axes[0]
    ax.set_xlabel("(E-E_F-{}meV ) (eV)".format(Ep*1000))
    ax.set_ylabel(r"DOS ($eV^{-1}$)")
    ax.axvline(0, 0, 10, color='red')
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim(0, 8)
    ax.legend()


    ax = axes[1]
    ax.plot(energy, -compute_TEP.dFD_dE(energy, 0, 300)/30, 'k--', label="-df/dE(T=300K)/30")
    ax.axvline(0, 0, 10, color='red')
    ax.set_xlabel("(E-E_F-{}meV ) (eV)".format(Ep*1000))
    ax.set_ylabel(r"TDF_{zz}")
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim(0, 1.5)
    ax.legend()


    ax = axes[2]
    ax.plot(energy, -compute_TEP.dFD_dE(energy, 0, 300)/50, 'k--', label="-df/dE(T=300K)/50")
    ax.axvline(0, 0, 10, color='red')
    ax.set_xlabel("(E-E_F-{}meV ) (eV)".format(Ep*1000))
    ax.set_ylabel(r"$TDF_{xx}$")
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim(0, 0.3)
    ax.legend()


    ax = axes[3]
    ax.set_xlabel("k-paths")
    ax.legend()

    tau_ratio = {
        "ib61" : 1.0,
        "ib62" : constC,
        "ib63" : constC,
        "ib64" : constC
    }

    onsgar_xx = None
    onsgar_zz = None

    for key in onsgar_dict.keys():
        if onsgar_xx is None:
            onsgar_xx = onsgar_dict[key]['xx']*tau_ratio[key]
            onsgar_zz = onsgar_dict[key]['zz']*tau_ratio[key]
            continue
        onsgar_xx += onsgar_dict[key]['xx']*tau_ratio[key]
        onsgar_zz += onsgar_dict[key]['zz']*tau_ratio[key]
        pass

    Szz = onsgar_zz[:,1]/onsgar_zz[:,0]/(-Tlist)
    axes[4].plot(Tlist, Szz*10**6,  'o', label="ib61 + ib(62,63,64)*{}".format(constC), color=colors[key])


    Sxx = onsgar_xx[:,1]/onsgar_xx[:,0]/(-Tlist)
    axes[5].plot(Tlist, Sxx*10**6, 'o', label="ib61 + ib(62,63,64)*{}".format(constC), color=colors[key])


    ax = axes[4]
    ax.set_ylim(-60, 20)
    ax.set_xlabel(r"Temperature (K)")
    ax.set_ylabel(r"Thermopwoer, $S_{zz}\, \mu V/K$")
    ax.legend()



    ax = axes[5]
    ax.set_xlabel(r"Temperature (K)")
    ax.set_ylabel(r"Thermopwoer, $S_{xx}\, \mu V/K$")
    ax.legend()


    ylimList = [
        (0,8),
        (0,1.3),
        (0,0.35),
        (-0.9,1.2),
        (-70,100),
        (-30, 150)
        ]

    for i, ax in enumerate(axes):
        ax.set_ylim(ylimList[i])
        pass

    plt.savefig("./NB3S4-band-resolved-CRTA-{}meV-shift-tau-ratio{}.png".format(Ep*1000, constC))




if __name__ == "__main__":
    """
    Best Strategy here is to copy compute_Szz_Sxx_v2 method and make edits in local scripts/notebooks,
    so that you can plot raw data as well and configure plots as needed.
    
    """
    # compute_energy_files("./data-file-schema.xml", out_dir="./energy", prefix="test", erange=(-0.1,0.1))
    plot_k_path(".", "./test.bt2", niter=2, unit_conv=Ha_to_eV)


    dft_data_dir = "./PBEsol-Relaxed/energy/ib61/"
    ylimList = [
        (0,8),
        (0,1),
        (0,0.2),
        (-1,1),
        (-160,220),
        (-30, 220)
        ]


    shift=48
    fig_name = "fig.png"
    fig, axes = compute_Szz_Sxx_v2(dft_data_dir, shift, ylimList, fig_name)



    pass




