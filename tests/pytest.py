
from dfttools import utils
from dfttools import mp_funcs
import matplotlib.pyplot as plt
import numpy as np

nx = 22
data_dir = "./tmp/"

# QE input fiels
nscf_path = data_dir + "nscf.{}.nb3s4.in".format(nx)
bands_path = data_dir + "bands_{}.nb3s4.in".format(nx)

# Data files
bands_data_path = data_dir + "bands_{}.dat".format(nx)
dos_data_path = data_dir + "tdos_{}.dat".format(nx)


data = utils.get_data_dict(nscf_path, bands_path, bands_data_path, dos_data_path)

# utils.plot_bands(data)

def Ry_to_ev(E):
    """
    Rydberg to ev
    1 Ry = 13.605693122990 ev
    """
    return E * 13.605693122990


with open("./tests/api_key", 'r') as f:
    api_key=f.readline()[:-1]
    print(len(api_key))
    pass

mp = mp_funcs.MPHelper(api_key)
material_id = "mp-12627"
mp.get_material_info(material_id)




# mp.plot_dos(sigma_=0.005)
# data = utils.get_data_dict(nscf_path, bands_path, bands_data_path, dos_data_path)
# x, y = data['dos'].T
# efermi = data['e_fermi']
# print(efermi)
# plt.plot(x-efermi, y, 'k--')
# plt.xlim(-2,3)
# plt.ylim(0,40)
# plt.show()





fig, axes = mp.plot_bands()
utils.plot_bands(data, "bands-combined.png", axesin=axes)

# # print(loaded_data.keys())
# ebands = data['bands']
# print("ebands.shape ", ebands.shape)
# branches = data['branches']
# efermi = data['e_fermi']
# # ebands = ebands[idx,]

# for i in range(9):
#     a , b = branches[i]['start_index'], branches[i]['end_index']
#     eb = ebands[:,a:b+1]

#     print(eb.shape)

#     x = np.linspace(0, 5, eb.shape[1])
#     # print("efermi ", loaded_data['e_fermi'])
#     # y = ebands.T - loaded_data['e_fermi']*(.997) # run 8
#     y = eb.T - efermi
#     # print(y.shape)
#     axes[i].plot(x, y, 'k--')


        

# plt.ylim(-1,1)
# plt.tight_layout() 
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=None)
# plt.savefig("bands-combined")

# plt.show()




def plot_dos_agains_mp(sigma):

    pass


