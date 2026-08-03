import numpy as np
import pickle
import matplotlib.pyplot as plt
import getpass


## Trying to make data parser that turns repeated steps into their own function
## Edit find_keywords function to add any other keyword classifications
## Run the parse_file on your file then do any necessary post-processing

def find_keywords(prev,line):
    #Assumes that keywords will be "&...", "#..." or follow empty space or "/"
    #If need to update formatting of keywords, change the if statements
    if any("&" in s for s in line):
        return True
    if any("#" in s for s in line):
        return True
    elif any("/" in s for s in prev):
        return True
    elif len(prev) == 0 and len(line) != 0:
        return True
    else:
        return False

#Helper function to count number of lines with same string length 
def get_data_length(i, lines):
    # print("get_data_length")
    start = i + 1
    while start < len(lines) and not lines[start].strip():
        start += 1

    if start >= len(lines):
        return [], start

    data = []
    j = start

    while j < len(lines):
        split_line = lines[j].strip().split()

        # if not split_line:  
        #     j += 1
        #     continue

        if len(split_line) != 0:
            data.append(split_line)
            j += 1
        else:
            break  
    # print(data)
    return data, j



def read_keywords(lines, keywords, include_keyword):
    data_blocks = {}
    i = 0
    #Will turn desired data into list of lists

    formatted_keywords = set(k.lower() for k in keywords)

    while i < len(lines):
        
        line = lines[i].strip()
        # print(line)
        vals = line.split()
        # print(vals)

        if find_keywords(lines[i-1].strip().split() if i > 0 else [], vals):
            formatted_vals = [v.lower().lstrip("&") for v in vals]
            found_keyword = formatted_keywords & set(formatted_vals)
            # print("found_keyword ", found_keyword)
            if found_keyword:
                keyword = list(found_keyword)[0]
                data, j = get_data_length(i, lines)
                if include_keyword:
                    # Store the keyword line as first entry
                    data = [vals] + data

                data_blocks[keyword] = data
                i = j
                continue 
        i += 1

    return data_blocks



def parse_file(file_path, keywords_list, include_keyword=False):
    """
    Method will parse the file inputted looking for all keywords that are in the
    keywords list. If include_keyword is true it will include the keyword line
    as the first entry in the data. 
    """
    # print("file_path ", file_path)
    with open(file_path, "r") as f:
        lines = f.readlines()

        file_data = read_keywords(lines, keywords_list, include_keyword)
    
    return file_data



def clean_up_branch_data(branch_list):
    """
    
    """
    print("clean_up_branch_data () *************")
    #Cleaning up branch_data
    branch_data_clean = {"branches":[]}
    gamma = 'Γ'
    #Looping through paths and stripping info
    prev_line = None
    index_counter = 0  
    name_list = []
    for line in branch_list:
        # print(line)
        if line[-1] == gamma:
            line[-1] = '\\Gamma'

        if prev_line is not None:
            segment_length = int(prev_line[-3])  
            # print("segment_length ", segment_length)

            if segment_length == 1: # means that the path doesn not contain any new path. already defined
                index_counter += 1
            if segment_length > 1:
                branch_dict = {
                    'name': f"{prev_line[-1]}-{line[-1]}",
                            'start_index': index_counter,
                            'end_index': index_counter + segment_length - 1
                        }
                        #print(branch_dict)
                # index_counter = branch_dict['end_index'] + 1
                index_counter += segment_length
                if branch_dict['name'] not in name_list:
                    name_list.append(branch_dict['name'])
                    branch_data_clean['branches'].append(branch_dict)
                else:
                    # print("path a 2nd time!")
                    pass
                
            else:
                # print("This was else")
                pass
        prev_line = line
        pass
    print("branch_data_clean ", branch_data_clean)
    return branch_data_clean


def clean_up_dos_data(dos_data_):
    #Cleaning dos & getting E-fermi#  
    dos_data_clean = {"dos":[], "e_fermi": None}
    for i, line in enumerate(dos_data_["e"]):
        if i == 0:
            dos_data_clean["e_fermi"] = float(line[-2])
        else:
            dos_data_clean["dos"].append(line[:2])
            pass
    dos_data_clean['dos'] = np.array(dos_data_clean['dos'], dtype=np.float32)
    return dos_data_clean


def parse_bands_file(filename):
    """
# To read bands.dat file generated by quantum espresso
# File format
# First line is the comment
# 2nd line is the first k-point -> a line with 3 entries
# All lines upto next k-point lines are energy of bands at previous k-point
    """
    nbands=0
    nks=0

    # we mark next line by these flags
    bandFlag = False
    kpointFlag = False

    kpoint_list = []
    eband_list = []
    band_block = []

    band_counter = 0
    with open(filename) as f:
        lines = f.readlines()
        # print(lines[0])
        head = ""
        for k, line in enumerate(lines):
            # print(line)
            if k==0:
                head = line.split()
                # print(head)
                nbands = int(head[2].split(',')[0])
                nks = int(head[4])
                # print(head)
                kpointFlag = True
                
            elif kpointFlag:
                kline = [float(a) for a in line.split()]
                
                kpoint_list.append(kline)
                kpointFlag = False
                bandFlag = True

            elif bandFlag:
                bline = [float(a) for a in line.split()]
                band_block += bline
                band_counter += len(bline)

                if band_counter >= nbands:
                    eband_list.append(band_block)
                    band_block = []
                    band_counter = 0
                    bandFlag = False
                    kpointFlag = True
                pass
            else:
                print("Logic failed.")
                pass
            pass
        pass


    ebands = np.array(eband_list).T
    kpoints = np.array(kpoint_list)

    # print(ebands.shape)
    # print(kpoints.shape)
    return {"bands":ebands, "kpoints":kpoints}


def get_data_dict(nscf_input_file, bands_input_file, bands_data_file, dos_data_file):
    """

    nscf_input_file  : nscf.in file  used for QE NSCF computation with pw.x command
    bands_input_file : bands.in file used for QE bands computation with pw.x command
    bands_data_file  : bands.dat file generated by bands post processing with command bands.x
    dos_data_file    : dos.dat file generted by dos post processing with command dos.x
    
    
    returns : dictionary with keys ['bands', 'kpoints', 'atomic_species', 'cell_parameters',
                 'atomic_positions', 'mesh_size', 'dos', 'e_fermi', 'branches']
    """

    #Retriving data from the 4 files
    nscf_data = parse_file(nscf_input_file, ["CELL_PARAMETERS", 
                                    "ATOMIC_SPECIES", 
                                    "ATOMIC_POSITIONS", 
                                    "K_POINTS"])
    
    # cell = nscf_data['cell_parameters']
    # print("lattice vector ", cell)

    # bands_data = parse_file(bands_path, ["plot"]) 
    branch_data = parse_file(bands_input_file, ["K_POINTS"]) 
    dos_data = parse_file(dos_data_file, ["E"], True) 


    #Post data processing

    #Splitting bands_data into kpoints and bands
    # bands_data_clean = {"kpoints":[], "bands":[]}
    # for line in bands_data["plot"]:
    #     if len(line) == 3:
    #         bands_data_clean["kpoints"].append(line)
    #     elif len(line) == 10:
    #         bands_data_clean["bands"].append(line)
    #     else: 
    #         continue
    
    
    # bands_data_clean["bands"] = np.array(bands_data_clean["bands"], dtype=np.float32).T
    # print("number of data", len(bands_data_clean["bands"]))
    # print(bands_data_clean['bands'][0])
    # print(bands_data_clean['bands'][1])
    #Renaming kpoints from nscf as mesh size
    nscf_data["mesh_size"] = nscf_data.pop("k_points")

    print(branch_data)
    branch_data_clean = clean_up_branch_data(branch_data["k_points"][1:])
    dos_data_clean = clean_up_dos_data(dos_data)
    #print(dos_data_clean["e_fermi"])

    bands_data_clean = parse_bands_file(bands_data_file)

    #Combine Data Together
    data = bands_data_clean|nscf_data|dos_data_clean|branch_data_clean

    # print(data["bands"].shape)
    return data

def parse_k_path(lines):
# The count line is right after the '/'
    namelist_end = 0
    count = int(lines[namelist_end + 1].strip())
    
    # Parse the k-point lines
    kpoint_lines = []
    for i in range(namelist_end + 2, namelist_end + 2 + count):
        line = lines[i].strip()
        if not line:
            continue
            
        parts = line.split('!')
        coords_nk = parts[0].split()
        label = parts[1].strip() if len(parts) > 1 else ''
        
        kx, ky, kz, nk = coords_nk[0], coords_nk[1], coords_nk[2], coords_nk[3]
        
        kpoint_lines.append({
            'kx': float(kx),
            'ky': float(ky),
            'kz': float(kz),
            'nk': int(nk),
            'label': label
        })
    
    # Build branch_data_clean exactly as in your example
    branch_data_clean = {"branches": []}
    gamma = 'Γ'
    prev_line = None
    index_counter = 0
    name_list = []
    
    for line in kpoint_lines:
        # Convert Γ for LaTeX/plotting compatibility
        label = line['label']
        if label == gamma:
            label = '\\Gamma'
        
        if prev_line is not None:
            segment_length = prev_line['nk']
            
            # segment_length == 1 means the path doesn't contain any new points
            if segment_length > 1:
                prev_label = prev_line['label']
                if prev_label == gamma:
                    prev_label = '\\Gamma'
                
                branch_dict = {
                    'name': f"{prev_label}-{label}",
                    'start_index': index_counter,
                    'end_index': index_counter + segment_length - 1
                }
                
                index_counter = branch_dict['end_index'] + 1
                
                if branch_dict['name'] not in name_list:
                    name_list.append(branch_dict['name'])
                    branch_data_clean['branches'].append(branch_dict)
        
        prev_line = line
    
    return branch_data_clean
    

def parse_matdyn(file_matdyn_in):
    """
    Parse a QE/qe-thermo input file containing k-point paths.

    File format:
    &input
        ...
    /
    <number_of_lines>
    kx ky kz nk ! label
    ...
    """
    with open(file_matdyn_in, 'r') as f:
        lines = f.readlines()
    
    # Find the '/' that ends the namelist
    namelist_end = None
    for i, line in enumerate(lines):
        if line.strip() == '/':
            namelist_end = i
            break
    
    if namelist_end is None:
        raise ValueError("Could not find end of namelist '/'")
    
    # The count line is right after the '/'
    branch = parse_k_path(lines[namelist_end+1:])
    print(branch)
    count = int(lines[namelist_end + 1].strip())
    
    # Parse the k-point lines
    kpoint_lines = []
    for i in range(namelist_end + 2, namelist_end + 2 + count):
        line = lines[i].strip()
        if not line:
            continue
            
        parts = line.split('!')
        coords_nk = parts[0].split()
        label = parts[1].strip() if len(parts) > 1 else ''
        
        kx, ky, kz, nk = coords_nk[0], coords_nk[1], coords_nk[2], coords_nk[3]
        
        kpoint_lines.append({
            'kx': float(kx),
            'ky': float(ky),
            'kz': float(kz),
            'nk': int(nk),
            'label': label
        })
    
    # Build branch_data_clean exactly as in your example
    branch_data_clean = {"branches": []}
    gamma = 'Γ'
    prev_line = None
    index_counter = 0
    name_list = []
    
    for line in kpoint_lines:
        # Convert Γ for LaTeX/plotting compatibility
        label = line['label']
        if label == gamma:
            label = '\\Gamma'
        
        if prev_line is not None:
            segment_length = prev_line['nk']
            
            # segment_length == 1 means the path doesn't contain any new points
            if segment_length > 1:
                prev_label = prev_line['label']
                if prev_label == gamma:
                    prev_label = '\\Gamma'
                
                branch_dict = {
                    'name': f"{prev_label}-{label}",
                    'start_index': index_counter,
                    'end_index': index_counter + segment_length - 1
                }
                
                index_counter = branch_dict['end_index'] + 1
                
                if branch_dict['name'] not in name_list:
                    name_list.append(branch_dict['name'])
                    branch_data_clean['branches'].append(branch_dict)
        
        prev_line = line
    
    return branch_data_clean



def get_phonon_data_dict(matdyn_input_file, freq_data_file, dos_data_file=None):
    """

    matdyn_input_file  : matdyn.in file used for QE phonon frequencies computation with matdyn.x command
    freq_data_file  : .freq file generated by matdyn.x
    dos_data_file    : dos.dat file generted by dos post processing with command dos.x. TODO
    
    
    returns : dictionary with keys ['bands', 'kpoints', 'dos', , 'branches']
    """

    branch_data = parse_matdyn(matdyn_input_file)
    bands_data = parse_bands_file(freq_data_file) 
    
    #Combine Data Together
    data = bands_data | branch_data

    # print(data["bands"].shape)
    return data


def plot_dos(data, axesin=None, fermi_factor=1.0):
    print(type(data['dos']))
    print(data['dos'].shape)
    x, y = data['dos'].T
    efermi = data['e_fermi']
    
    if abs(fermi_factor - 1.0) > 1e-3:
        print("Scaling fermi energy")
        axesin.set_xlabel(r"$E-E_F (eV), E_F={} * {}eV$".format(efermi, fermi_factor))
        efermi *= fermi_factor
        pass

    axesin.plot(x-efermi, y)
    axesin.set_ylabel(r"$??$")


    pass

def plot_bands(data, axesin=None, fermi_factor=1.0, symbol_='k--', legend_label=None):
    """
    loaded_data : data dictionary
    axesin : matplotlib axis
    fermi_factor : multiplicative factor to fermi energy to shift the bands
    """
    print("plotting bands")
    branches=data['branches']
    # branches

    modified_branch = []
    namelist = []
    for b in branches:
        length = b['end_index'] - b['start_index']
        if b['name'] in namelist or length <= 1:
            continue
        else:
            namelist.append(b['name'])
            modified_branch.append(b)
            pass
    branches = modified_branch
    print("branches ", branches)
    width_ratios = list(map(lambda x: x['end_index']-x['start_index'], branches))
    # print(width_ratios)
    if axesin is None:
        fig, axes = plt.subplots(1, len(branches), figsize=(10, 6), sharey=True, gridspec_kw={"width_ratios": width_ratios}, dpi=200)
    else:
        axes = axesin
    if len(branches) != len(axes):
        print("branch size and axes count does not match!! <<Warning>>")
        pass

    
        
    # print(loaded_data.keys())
    ebands = data['bands']
    print("ebands.shape ", ebands.shape)
    efermi = data['e_fermi']
    # ebands = ebands[idx,]
    axes[0].set_ylabel(r"$E-E_F (eV), E_F={} eV$".format(efermi))
    if abs(fermi_factor - 1.0) > 1e-3:
        print("Scaling fermi energy")
        axes[0].set_ylabel(r"$E-E_F (eV), E_F={} * {}eV$".format(efermi, fermi_factor))
        pass

    for i in range(len(branches)):
        a , b = branches[i]['start_index'], branches[i]['end_index']
        eb = ebands[:,a:b+1]

        print(eb.shape)

        x = np.linspace(0, 5, eb.shape[1])
        print("efermi ", efermi, " factor ", fermi_factor)
        y = eb.T - efermi*fermi_factor
        # print(y.shape)
        if i == len(branches)-1:
            label_line = axes[i].plot(x, y, symbol_)
        else:
            axes[i].plot(x, y, symbol_)
            pass
        axes[i].set_xlabel(r"${}$".format(branches[i]['name']), fontsize=14)
        axes[i].set_xlim(x[0], x[-1])
        axes[i].set_xticks([])
    
    # plt.ylim(-1,1)
    # plt.tight_layout() 
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=None)
    # plt.savefig(filename)
    return label_line


def read_dos_input_file():
    """
    TODO
    """


    pass


if __name__=="__main__":
    username = getpass.getuser()

    #File inputs and naming output file
    # signature = "nb3s4-run8"
    # folder_path = "/home/{}/quantum-espresso/".format(username) + "/"
    # bands_path = folder_path + "bands.dat"
    # dos_path = folder_path + "dos.dat"
    # nscf_path = folder_path + "input/nscf.in"
    # branches_path = folder_path + "input/bands.in"

    # output_dir = "/home/{}/materials-project/".format(username)
    # filename = output_dir + "QE-{}.pkl".format(signature)

    # data = get_data_dict(nscf_path, bands_path, branches_path, dos_path)

    #Saving pickle file
    # with open(filename, 'wb') as f:
    #     pickle.dump(data, f)
    #     print(f"Pickled data saved to: {filename}")
 

    nx = 22
    data_dir = "/Users/shahnoor/projects/dfttools/tmp/"
    bands_path = data_dir + "bands_{}.dat".format(nx)
    dos_path = data_dir + "dos_{}.dat".format(nx)
    nscf_path = data_dir + "nscf.{}.nb3s4.in".format(nx)
    branches_path = data_dir + "bands_{}.nb3s4.in".format(nx)
    data = get_data_dict(nscf_path, branches_path, bands_path, dos_path)

    plot_bands(data)
    # filename = "test_data{}.pkl".format(nx)
    # #Saving pickle file
    # with open(filename, 'wb') as f:
    #     pickle.dump(data, f)
    #     print(f"Pickled data saved to: {filename}")

    # pass

