import numpy as np


def generate_k_points(nx, ny, nz):
    print("K points block for .win file")
    print("mp_grid = {} {} {}".format(nx, ny, nz))
    print("begin kpoints")
    for kx in np.linspace(0, 1, nx, endpoint=False):
        for ky in np.linspace(0, 1, ny, endpoint=False):
            for kz in np.linspace(0, 1, nz, endpoint=False):
                print("{:.4f}  {:.4f}  {:.4f}".format(kx, ky, kz))
                pass
            pass
    print("end kpoints")


def k_points_nscf(nx, ny, nz):
    total = nx * ny * nz * 1.0
    print(r"K_POINTS {crystal}")
    print(total)
    for kx in np.linspace(0, 1, nx, endpoint=False):
        for ky in np.linspace(0, 1, ny, endpoint=False):
            for kz in np.linspace(0, 1, nz, endpoint=False):
                print("{:.4f}  {:.4f}  {:.4f} {:.4f}".format(kx, ky, kz, 1.0/total))
                pass