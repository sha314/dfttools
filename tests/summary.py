
from dfttools import utils
from dfttools import mp_funcs

with open("./api_key", 'r') as f:
    api_key=f.readline()[:-1]
    print(len(api_key))
    pass

mp = mp_funcs.MPHelper(api_key)
material_id = "mp-12627" # Nb3S4, hexagonal
# material_id = "mp-135" # for Li bcc
# material_id = "mp-21008" # for cubic Ni3Ge
# material_id = "mp-561" # for Hexagonal Nb3Se4
# material_id = "mp-134" # for cubic Al
# material_id = "mp-169" # for Graphene monolayer

# material_id = "mp-66" # for diamond

material_id = "mp-13" # for diamond

mp.get_material_info(material_id)

mp.get_band_k_paths(1)
mp.get_atomic_masses()
mp.get_atomic_positions()
mp.get_Cell_parameters()
mp.approx_mesh_size(delta_k=0.03)
mp.get_band_count()


# Create a folder in "./tmp" named by material id
# same the info as text file
# save Bands, DoS, Bands+Dos as png image
# mp.plot_bands_dos_png("./tmp/")

