



import numpy as np

# KIMI.AI
def rotate_k_fs(xk, nks, nsym, s, time_reversal, t_rev, nspin,
                at, ef, ef_up, ef_dw, two_fermi_energies,
                k1, k2, k3, nk1, nk2, nk3, nbnd, et, stdout=None):
    """
    Python equivalent of QE/PP/src/fermisurfer_common.f90::rotate_k_fs
    
    Finds the equivalent k-point in the irreducible BZ for the whole BZ
    and computes the min/max band indices containing Fermi surfaces.
    
    Parameters
    ----------
    xk : ndarray, shape (3, nks)
        K-points in Cartesian coordinates (units of 2π/alat).
    nks : int
        Total number of k-points (times spin if nspin==2).
    nsym : int
        Number of symmetries.
    s : ndarray, shape (3, 3, nsym)
        Symmetry matrices in crystal coordinates (integer).
    time_reversal : bool
        Global time-reversal flag.
    t_rev : ndarray, shape (nsym,)
        Time-reversal flag per symmetry (1 if includes time reversal).
    nspin : int
        1 or 2.
    at : ndarray, shape (3, 3)
        Direct lattice vectors (in alat units).
    ef, ef_up, ef_dw : float
        Fermi energies.
    two_fermi_energies : bool
        Whether spin-polarized calculation has two Fermi levels.
    k1, k2, k3 : int
        K-point grid offsets.
    nk1, nk2, nk3 : int
        K-point grid dimensions.
    nbnd : int
        Number of bands.
    et : ndarray, shape (nbnd, nks)
        Eigenvalues.
    stdout : file-like, optional
        Output stream for logging (default: sys.stdout).
    
    Returns
    -------
    equiv : ndarray, shape (nk1, nk2, nk3)
        Maps each grid point (0-based indices) to the irreducible k-point
        index (0-based).  NOTE: Fortran original uses 1-based indices.
    b_low : int
        Lowest band containing Fermi surface (0-based band index).
    b_high : int
        Highest band containing Fermi surface (0-based band index).
    """
    import sys
    if stdout is None:
        stdout = sys.stdout

    # --- Band range containing Fermi level ---
    if two_fermi_energies:
        ef1 = ef_up
        ef2 = ef_dw
    else:
        ef1 = ef
        ef2 = ef

    # Initialize with safe defaults (0-based in Python)
    b_low = nbnd - 1
    b_high = 0

    for ibnd in range(nbnd):
        # ibnd is 0-based here; et shape is (nbnd, nks)
        if np.min(et[ibnd, :nks]) < max(ef1, ef2):
            b_high = ibnd
        if np.max(et[nbnd - 1 - ibnd, :nks]) > min(ef1, ef2):
            b_low = nbnd - 1 - ibnd

    print("", file=stdout)
    print(f"     Number of bands : {nbnd}", file=stdout)
    print(f"     Number of k times spin : {nks}", file=stdout)
    print(f"     Number of symmetries : {nsym}", file=stdout)
    print(f"     Lowest band which contains FS : {b_low + 1}", file=stdout)
    print(f"     Highest band which contains FS : {b_high + 1}", file=stdout)

    # --- Build equivalence map ---
    equiv = np.zeros((nk1, nk2, nk3), dtype=int)
    ldone = np.zeros((nk1, nk2, nk3), dtype=bool)

    if nspin == 2:
        nk = nks // 2
    else:
        nk = nks

    grid_size = np.array([nk1, nk2, nk3], dtype=int)
    print("grid_size ", grid_size)
    k_offset = np.array([k1, k2, k3], dtype=float)
    half_offset = 0.5 * k_offset

    if xk.shape[0] != 3:
        xk = xk.T
        pass

    for ik in range(nk):
        # Convert Cartesian k-point to fractional/crystal coordinates
        # Fortran: matmul(xk(1:3,ik), at(1:3,1:3))
        xk_frac = xk[:, ik] @ at  # shape (3,)
        if ik == 4:
            break
        for isym in range(nsym):
            # Apply symmetry in crystal coords, then scale to grid
            # Fortran: MATMUL(REAL(s(1:3,1:3,isym), DP), xk_frac(1:3)) * (/nk1, nk2, nk3/)
            mat = s[:, :, isym].astype(np.float64)
            print("mat ", mat)
            kv = (mat @ xk_frac) * grid_size.astype(np.float64)

            # Time reversal for this symmetry
            if t_rev[isym] == 1:
                kv = -kv

            # Shift by offset
            print("kv ", kv)
            kv = kv - half_offset
            print("kv ", kv)

            # Round to nearest grid point

            ikv = np.rint(kv).astype(int)

            # Check if this maps exactly to a grid point
            threshold = 1e-8
            if np.any(np.abs(kv - ikv.astype(np.float64)) > threshold):
                continue
            print("passed threshold ", threshold)

            # Wrap to grid (Fortran MODULO -> Python % for positive modulus)
            print("ikv ", ikv)
            ikv = ikv % grid_size  # gives 0 .. nk-1
            print(ikv)

            equiv[ikv[0], ikv[1], ikv[2]] = ik
            ldone[ikv[0], ikv[1], ikv[2]] = True

            # Global time-reversal
            if time_reversal:
                # Fortran: ikv = -(ikv - 1) - (/k1, k2, k3/)
                # In 0-based: ikv_new = -(ikv) - k_offset
                ikv_tr = -(ikv) - k_offset.astype(int)
                ikv_tr = ikv_tr % grid_size

                equiv[ikv_tr[0], ikv_tr[1], ikv_tr[2]] = ik
                ldone[ikv_tr[0], ikv_tr[1], ikv_tr[2]] = True

    # --- Check coverage ---
    not_done = np.count_nonzero(~ldone)
    if not_done != 0:
        print(f"  # of elements that are not done : {not_done}", file=stdout)

    return equiv, b_low, b_high



# ChatGPT


# def rotate_k_fs(
#     xk,
#     at,
#     s,
#     t_rev,
#     time_reversal,
#     nk1,
#     nk2,
#     nk3,
#     k1,
#     k2,
#     k3,
#     et,
#     ef,
#     *,
#     ef_up=None,
#     ef_dw=None,
#     two_fermi_energies=False,
#     nspin=1,
# ):
#     """
#     Python equivalent of Quantum ESPRESSO's rotate_k_fs().

#     Parameters
#     ----------
#     xk : ndarray, shape (3, nks)
#         k-points in Cartesian coordinates, as in QE's xk.
#     at : ndarray, shape (3, 3)
#         Direct-lattice matrix, as in QE.
#     s : ndarray, shape (3, 3, nsym)
#         Symmetry matrices.
#     t_rev : ndarray, shape (nsym,)
#         1 if the corresponding symmetry operation includes
#         time reversal, otherwise 0.
#     time_reversal : bool
#         QE's global time_reversal flag.
#     nk1, nk2, nk3 : int
#         Dimensions of the full k-point grid.
#     k1, k2, k3 : int
#         Monkhorst-Pack offsets used by QE.
#     et : ndarray, shape (nbnd, nks)
#         Band energies.
#     ef : float
#         Fermi energy.
#     ef_up, ef_dw : float, optional
#         Spin-up/down Fermi energies when two_fermi_energies=True.
#     two_fermi_energies : bool
#         Corresponds to QE's two_fermi_energies.
#     nspin : int
#         Number of spin channels.

#     Returns
#     -------
#     equiv : ndarray, shape (nk1, nk2, nk3), dtype=int
#         Fortran-style k-point index (1-based), matching QE's equiv.
#         Entries therefore range from 1 through nk.
#     b_low : int
#         Lowest band containing a Fermi surface.
#     b_high : int
#         Highest band containing a Fermi surface.
#     """

#     xk = np.asarray(xk)
#     at = np.asarray(at)
#     s = np.asarray(s)
#     t_rev = np.asarray(t_rev)
#     et = np.asarray(et)

#     nbnd, nks = et.shape
#     nsym = s.shape[2]

#     # ------------------------------------------------------------
#     # Which bands contain the Fermi level?
#     #
#     # Direct translation of:
#     #
#     # IF(MINVAL(et(ibnd,1:nks)) < MAX(ef1,ef2)) b_high = ibnd
#     # IF(MAXVAL(et(nbnd-ibnd+1,1:nks)) > MIN(ef1,ef2))
#     #                                      b_low = nbnd-ibnd+1
#     # ------------------------------------------------------------

#     if two_fermi_energies:
#         if ef_up is None or ef_dw is None:
#             raise ValueError(
#                 "ef_up and ef_dw are required when "
#                 "two_fermi_energies=True"
#             )
#         ef1 = ef_up
#         ef2 = ef_dw
#     else:
#         ef1 = ef
#         ef2 = ef

#     b_low = None
#     b_high = None

#     for ibnd0 in range(nbnd):
#         # Fortran ibnd = ibnd0 + 1
#         if np.min(et[ibnd0, :]) < max(ef1, ef2):
#             b_high = ibnd0 + 1

#         # Fortran: nbnd - ibnd + 1
#         # Python zero-based:
#         band = nbnd - ibnd0 - 1

#         if np.max(et[band, :]) > min(ef1, ef2):
#             b_low = band + 1

#     if b_low is None or b_high is None:
#         raise RuntimeError(
#             "Could not identify bands containing the Fermi surface."
#         )

#     # Number of unique k-points before spin duplication.
#     if nspin == 2:
#         nk = nks // 2
#     else:
#         nk = nks

#     # Fortran:
#     #   equiv(nk1,nk2,nk3)
#     #   ldone(nk1,nk2,nk3)
#     #
#     # Use zero-based Python indexing internally.
#     equiv = np.full((nk1, nk2, nk3), -1, dtype=np.int64)
#     ldone = np.zeros((nk1, nk2, nk3), dtype=bool)

#     grid = np.array([nk1, nk2, nk3], dtype=float)
#     kshift = np.array([k1, k2, k3], dtype=float)

#     # ------------------------------------------------------------
#     # Main mapping
#     # ------------------------------------------------------------

#     for ik0 in range(nk):
#         # Fortran:
#         #
#         # xk_frac = matmul(xk(:,ik), at)
#         #
#         # IMPORTANT:
#         # NumPy's row-vector operation x @ at is the same
#         # index contraction as the Fortran MATMUL here.
#         xk_frac = xk[:, ik0] @ at

#         for isym in range(nsym):

#             # Fortran:
#             #
#             # kv = MATMUL(REAL(s(:,:,isym),DP), xk_frac)
#             #      * (/nk1,nk2,nk3/)
#             #
#             # With NumPy column-vector convention:
#             kv = s[:, :, isym].astype(float) @ xk_frac
#             kv *= grid

#             # Symmetry operation may itself contain time reversal.
#             if t_rev[isym] == 1:
#                 kv = -kv

#             # Fortran:
#             #
#             # kv = kv - 0.5 * (/k1,k2,k3/)
#             kv -= 0.5 * kshift

#             # NINT in QE/Fortran is nearest integer.
#             # np.rint uses round-to-even at exact half integers,
#             # so use an explicit implementation if exact boundary
#             # behavior matters.
#             ikv = np.floor(kv + 0.5).astype(np.int64)

#             # Fortran:
#             #
#             # IF(ANY(ABS(kv - REAL(ikv)) > 1d-8)) CYCLE
#             if np.any(np.abs(kv - ikv) > 1.0e-8):
#                 continue

#             # Fortran:
#             #
#             # ikv = MODULO(ikv, (/nk1,nk2,nk3/)) + 1
#             #
#             # Convert immediately to zero-based Python indices.
#             ikv = np.mod(ikv, [nk1, nk2, nk3])

#             # Fortran equiv stores the 1-based irreducible k-point.
#             equiv[ikv[0], ikv[1], ikv[2]] = ik0 + 1
#             ldone[ikv[0], ikv[1], ikv[2]] = True

#             # ----------------------------------------------------
#             # Global time reversal
#             # ----------------------------------------------------

#             if time_reversal:
#                 # Fortran:
#                 #
#                 # ikv = - (ikv - 1) - (/k1,k2,k3/)
#                 #
#                 # Here ikv is already zero-based.
#                 ikv = -ikv - np.array([k1, k2, k3])

#                 ikv = np.mod(ikv, [nk1, nk2, nk3])

#                 equiv[ikv[0], ikv[1], ikv[2]] = ik0 + 1
#                 ldone[ikv[0], ikv[1], ikv[2]] = True

#     # Equivalent of:
#     #
#     # COUNT(.NOT. ldone)
#     #
#     missing = np.count_nonzero(~ldone)

#     if missing:
#         print(f" # of elements that are not done : {missing}")

#     return equiv, b_low, b_high