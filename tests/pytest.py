
from dfttools import utils


nx = 22
data_dir = "./tests/"

# QE input fiels
nscf_path = data_dir + "nscf.{}.nb3s4.in".format(nx)
bands_path = data_dir + "bands_{}.nb3s4.in".format(nx)

# Data files
bands_data_path = data_dir + "bands_{}.dat".format(nx)
dos_data_path = data_dir + "tdos_{}.dat".format(nx)


data = utils.get_data_dict(nscf_path, bands_path, bands_data_path, dos_data_path)

utils.plot_bands(data)


