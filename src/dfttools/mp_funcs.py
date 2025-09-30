from mp_api.client import MPRester
import math
from pymatgen.core import Element
from pymatgen.ext.matproj import MPRester
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter, BSDOSPlotter
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.core import Spin
import numpy as np


class MPHelper:
    def __init__(self, api_key):
        self.api_key = api_key
        pass

    def get_material_info(self, material_id):
         with MPRester(self.api_key) as mpr:
            self.structure = mpr.get_structure_by_material_id(material_id)
            self.bs = mpr.get_bandstructure_by_material_id(material_id, line_mode=True)
            self.dos = mpr.get_dos_by_material_id(material_id)
            # Search by material ID or formula
            self.summary = mpr.summary.search(material_ids=material_id)

            self.docs = mpr.materials.search(material_ids=[material_id])
            pass

    def approx_mesh_size(self, delta_k=0.08):
        """
        delta_k = 0.08  # (1/Å)
        """
        

        a,b,c = self.structure.lattice.abc

        



        """Compute Monkhorst-Pack mesh sizes based on lattice constants and desired Δk (1/Å)."""
        kx = math.ceil((2 * math.pi) / (a * delta_k))
        ky = math.ceil((2 * math.pi) / (b * delta_k))
        kz = math.ceil((2 * math.pi) / (c * delta_k))

        print(f"Lattice parameters: a = {a} Å, b = {b} Å, c = {c} Å")
        print(f"Target Δk = {delta_k:.3f} Å⁻¹")
        print(f"Recommended k-point mesh: {kx} × {ky} × {kz}")
        print("Total points ", kx * ky * kz)

    def get_atomic_masses(self):
        elements = sorted(set(site.specie.symbol for site in self.structure))

        print("ATOMIC_SPECIES")
        for el in elements:
            atomic_mass = Element(el).atomic_mass
            print(f"  {el}  {atomic_mass:.3f}")

            pass

    def get_Cell_parameters(self):
        print("CELL_PARAMETERS angstrom")
        for vector in self.structure.lattice.matrix:
            print(f"  {vector[0]:.6f}  {vector[1]:.6f}  {vector[2]:.6f}")
            pass

    def get_atomic_positions(self):
        # Print fractional atomic positions (QE uses these for crystal units)
        print("ATOMIC_POSITIONS {crystal}")
        for site in self.structure:
            element = site.species_string
            x, y, z = site.frac_coords
            print(f" {element:2} {x:.7f} {y:.7f} {z:.7f}")
            pass

    def plot_dos(self, sigma_):
        
        ## Plotting DOS
        plotter_dos = DosPlotter(sigma=sigma_)
        plotter_dos.add_dos("Total dos", self.dos)
        plotter_dos.get_plot()

    def plot_bands(self):
        width_ratios = list(map(lambda x: x['end_index']-x['start_index'], self.bs.branches))
        # print(width_ratios)
        fig, axes = plt.subplots(1, len(self.bs.branches), figsize=(10, 6), sharey=True, gridspec_kw={"width_ratios": width_ratios}, dpi=200)

        axes[0].set_ylabel(r"$E-E_F (eV)$")

        for i in range(9):
            jj = i
            kk = i
            
            ebands = self.bs.bands[Spin.up]
            print(ebands.shape)

            
            a , b = self.bs.branches[kk]['start_index'], self.bs.branches[kk]['end_index']
            ebands = ebands[:,a:b+1]
            print(ebands.shape)
            x = np.linspace(0, 5, ebands.shape[1])
            y = ebands.T - self.bs.efermi
            # print(y.shape)
            axes[i].plot(x, y, 'r-')
            axes[i].set_xlabel(r"${}$".format(self.bs.branches[jj]['name']))
            axes[i].set_xlim(x[0], x[-1])
            axes[i].set_xticks([])


        return fig, axes


    def get_dos(self):
        return self.dos
    
    def get_bands(self):
        return self.bs
    


