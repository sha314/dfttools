import numpy as np

from dfttools.units import *

def read_frmsf(filename):
    """
    
    
    """
        
    with open(filename) as f:
        lines = f.readlines()
        # print(lines[1].split())
        nk = nk1, nk2, nk3 = [int(x) for x in lines[0].split()]
        nspin = int(lines[1].split()[0]) # maybe it's the shift and not the spin
        nband = int(lines[2].split()[0])
        b1 = [float(x) for x in lines[3].split()]
        b2 = [float(x) for x in lines[4].split()]
        b3 = [float(x) for x in lines[5].split()]
        tmp = []
        for i in range(6, len(lines)):
            tmp.append(float(lines[i]))
            pass
        pass
    bands = np.zeros((nband, nk1, nk2, nk3))
    offset = np.prod((nband, nk1, nk2, nk3))
    color_matrix = np.zeros((nband, nk1, nk2, nk3))
    for iband in range(nband):
        for i1 in range(nk1):
            for i2 in range(nk2):
                for i3 in range(nk3):
                    idx = iband * nk1 * nk2 * nk3 \
                                 + i1 * nk2 * nk3 \
                                       + i2 * nk3 + i3
                    bands[iband, i1, i2, i3] = tmp[idx]
                    color_matrix[iband, i1, i2, i3] = tmp[offset + idx]
                    pass
                pass
            pass
        pass

    
    
    bvec = np.vstack([b1, b2, b3])


    result = {
        'nk': nk,
        'ishift': 0,
        'nspin': nspin,
        'nband': nband,
        'bvec': bvec,
        'energy': bands,
        'scalar': color_matrix,
        'n_scalar_blocks': len(color_matrix)
    }
    return result
    


def write_frmsf(filename, result, transformEnergy=None, transformColor=None):
    """
    
    Writes the data in fermisurfer format and applies transform method(s).

    transformEnergy: method to do unit conversion of energy
    transformColor : method to color data, maybe taking absolute value or doing unit conversion 
    """
    nk = nk1, nk2, nk3 = result['nk']
    nband = result['nband']
    bvec = result['bvec']
    nspin = result['nspin']
    b1, b2, b3 = bvec[0], bvec[1], bvec[2]
    bands = result['energy']
    color_matrix = result['scalar']

    if transformColor is None:
        transformColor = lambda x: x
    if transformEnergy is None:
        transformEnergy = lambda x: x


    lines = []
    lines.append(f"{nk1}\t{nk2}\t{nk3}\n")
    lines.append(f"{nspin}\n")
    lines.append(f"{nband}\n")
    lines.append(f"{b1[0]}\t{b1[1]}\t{b1[2]}\n")
    lines.append(f"{b2[0]}\t{b2[1]}\t{b2[2]}\n")
    lines.append(f"{b3[0]}\t{b3[1]}\t{b3[2]}\n")


    offset = np.prod((nband, nk1, nk2, nk3))
    tmp = [0]*offset*2
    for iband in range(nband):
        for i1 in range(nk1):
            for i2 in range(nk2):
                for i3 in range(nk3):
                    idx = iband * nk1 * nk2 * nk3 \
                                 + i1 * nk2 * nk3 \
                                       + i2 * nk3 + i3
                    # print("idx ", idx)
                    tmp[idx] = transformEnergy(bands[iband, i1, i2, i3])
                    tmp[offset + idx] = transformColor(color_matrix[iband, i1, i2, i3])
                    pass
                pass
            pass
        pass
    for a in tmp:
        lines.append(f"{a}\n")
        pass

    with open(filename, 'w') as f:
        f.writelines(lines)
            
    pass


    


def compute_effective_mass(files):
    """
    effective mass (m*)_ab is in
    """

    result_mxx = result_imxx = read_frmsf (files["mxx"])
    result_myy = result_imyy = read_frmsf (files["myy"])
    result_mzz = result_imzz = read_frmsf (files["mzz"])
    result_mxy = result_imxy = read_frmsf (files["mxy"])
    result_mxz = result_imxz = read_frmsf (files["mxz"])
    result_myz = result_imyz = read_frmsf (files["myz"])

    nk1, nk2, nk3 = result_imxx['nk']
    nband = result_imxx['nband']

    # 6 independent elements at each k point
    imxx = result_imxx['scalar']
    imyy = result_imyy['scalar']
    imzz = result_imzz['scalar']
    imxy = result_imxy['scalar']
    imxz = result_imxz['scalar']
    imyz = result_imyz['scalar']

    
    # we need to compute mass matrix inverse at each k point

    mxx = np.zeros((nband, nk1, nk2, nk3))
    myy = np.zeros((nband, nk1, nk2, nk3))
    mzz = np.zeros((nband, nk1, nk2, nk3))
    mxy = np.zeros((nband, nk1, nk2, nk3))
    mxz = np.zeros((nband, nk1, nk2, nk3))
    myz = np.zeros((nband, nk1, nk2, nk3))

    for iband in range(nband):
        for i1 in range(nk1):
            for i2 in range(nk2):
                for i3 in range(nk3):
                    IMat = np.array([
                        [imxx[iband, i1, i2, i3], imxy[iband, i1, i2, i3], imxz[iband, i1, i2, i3]],
                        [imxy[iband, i1, i2, i3], imyy[iband, i1, i2, i3], imyz[iband, i1, i2, i3]],
                        [imxz[iband, i1, i2, i3], imyz[iband, i1, i2, i3], imzz[iband, i1, i2, i3]]
                    ])
                    try:
                        Mat = np.linalg.inv(IMat)
                    except np.linalg.LinAlgError:
                        print(f"np.linalg.LinAlgError at [{i1},{i2},{i3}]")
                        Mat = np.full((3, 3), np.nan)
                        pass

                    Mat = np.nan_to_num(Mat)
                    if abs(Mat[0,0]) > 20:
                        print(f"Too large mass at [{i1},{i2},{i3}]")
                        print(Mat)
                    mxx[iband, i1, i2, i3] = Mat[0,0]
                    myy[iband, i1, i2, i3] = Mat[1,1]
                    mzz[iband, i1, i2, i3] = Mat[2,2]
                    mxy[iband, i1, i2, i3] = Mat[0,1]
                    mxz[iband, i1, i2, i3] = Mat[0,2]
                    myz[iband, i1, i2, i3] = Mat[1,2]
                    pass
                pass
            pass
        pass
    
    result_mxx['scalar'] = mxx
    result_myy['scalar'] = myy
    result_mzz['scalar'] = mzz
    result_mxy['scalar'] = mxy
    result_mxz['scalar'] = mxz
    result_myz['scalar'] = myz

    return result_mxx, result_myy, result_mzz, result_mxy, result_mxz, result_myz


def get_effective_mass():
    DATA_PATH = "./FermiSurferInputs/qe/"
    DATA_PATH = "./"
    result_mxx, result_myy, result_mzz, result_mxy, result_mxz, result_myz = compute_effective_mass(
        {
            "mxx" : DATA_PATH + "nb3s4_inv_mstar_xx.frmsf",
            "myy" : DATA_PATH + "nb3s4_inv_mstar_yy.frmsf",
            "mzz" : DATA_PATH + "nb3s4_inv_mstar_zz.frmsf",
            "mxy" : DATA_PATH + "nb3s4_inv_mstar_xy.frmsf",
            "mxz" : DATA_PATH + "nb3s4_inv_mstar_xz.frmsf",
            "myz" : DATA_PATH + "nb3s4_inv_mstar_yz.frmsf"
        }
    )

    write_frmsf(DATA_PATH + "nb3s4_mstar_xx.frmsf", result_mxx)
    write_frmsf(DATA_PATH + "nb3s4_mstar_yy.frmsf", result_myy)
    write_frmsf(DATA_PATH + "nb3s4_mstar_zz.frmsf", result_mzz)
    write_frmsf(DATA_PATH + "nb3s4_mstar_xy.frmsf", result_mxy)
    write_frmsf(DATA_PATH + "nb3s4_mstar_xz.frmsf", result_mxz)
    write_frmsf(DATA_PATH + "nb3s4_mstar_yz.frmsf", result_myz)
    pass


def modify_fermi_velocity():
    DATA_PATH = "./FermiSurferInputs/qe/"


    filename = DATA_PATH + "nb3s4_vfermi.frmsf"
    result = read_frmsf (filename)

    filename = DATA_PATH + "nb3s4_vfermi-test.frmsf"
    write_frmsf(filename, result, transformEnergy=lambda x: x*Ry_to_eV, transformColor=abs)

    filename = DATA_PATH + "nb3s4_vfermi_x.frmsf"
    result = read_frmsf (filename)

    filename = DATA_PATH + "nb3s4_vfermi_x-test.frmsf"
    write_frmsf(filename, result, transformEnergy=lambda x: x*Ry_to_eV, transformColor=abs)


    filename = DATA_PATH + "nb3s4_vfermi_y.frmsf"
    result = read_frmsf (filename)

    filename = DATA_PATH + "nb3s4_vfermi_y-test.frmsf"
    write_frmsf(filename, result, transformEnergy=lambda x: x*Ry_to_eV, transformColor=abs)


    filename = DATA_PATH + "nb3s4_vfermi_z.frmsf"
    result = read_frmsf (filename)

    filename = DATA_PATH + "nb3s4_vfermi_z-test.frmsf"
    write_frmsf(filename, result, transformEnergy=lambda x: x*Ry_to_eV, transformColor=abs)



def main():
    # modify_fermi_velocity()
    get_effective_mass()

    pass


if __name__ == "__main__":
    main()
    pass