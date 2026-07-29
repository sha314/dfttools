


def plot_bands_v3(data, axesin=None, fermi_shift=0.0, symbol_='k--', legend_label=None, branches=None, xticks=None, yticks=None):
    """
    loaded_data : data dictionary
    axesin : matplotlib axis
    fermi_factor : multiplicative factor to fermi energy to shift the bands
    """
    print("plotting bands")
    if branches is None:
        branches_b=data['branches']
    else:
        branches_b=branches
    # branches

    modified_branch = []
    namelist = []
    for b in branches_b:
        length = b['end_index'] - b['start_index']
        if b['name'] in namelist or length <= 1:
            continue
        else:
            namelist.append(b['name'])
            modified_branch.append(b)
            pass
    branches_b = modified_branch
    print("branches ", branches_b)
    width_ratios = list(map(lambda x: x['end_index']-x['start_index'], branches_b))
    # print(width_ratios)
    if axesin is None:
        fig, axes = plt.subplots(1, len(branches_b), figsize=(10, 6), sharey=True, gridspec_kw={"width_ratios": width_ratios}, dpi=200)
    else:
        axes = axesin
    if len(branches_b) != len(axes):
        print("branch size and axes count does not match!! <<Warning>>")
        pass

    
        
    # print(loaded_data.keys())
    ebands = data['bands']
    print("ebands.shape ", ebands.shape)
    efermi = data['e_fermi']
    # ebands = ebands[idx,]
    axes[0].set_ylabel(r"$E-E_F (eV), E_F={} eV$".format(efermi))
    if abs(fermi_shift - 0.0) > 1e-3:
        print("shifting fermi energy")
        axes[0].set_ylabel(r"$E-E_F (eV), E_F={} - {}eV$".format(efermi, fermi_shift))
        pass

    for i in range(len(branches_b)):
        a , b = branches_b[i]['start_index'], branches_b[i]['end_index']
        name = branches_b[i]['name']
        eb = ebands[:,a:b+1]

        print(eb.shape)

        x = np.linspace(0, 1, eb.shape[1])
        print("efermi ", efermi, " factor ", fermi_shift)
        y = eb.T - efermi-fermi_shift
        # print(y.shape)
        if i == len(branches_b)-1:
            label_line = axes[i].plot(x, y, symbol_)
        else:
            axes[i].plot(x, y, symbol_)
            pass
        # axes[i].set_xlabel(r"${}$".format(branches_b[i]['name']), fontsize=14)

        axes[i].set_xlim(x[0], x[-1])
        if xticks is not None:
            axes[i].set_xticks(xticks)
        if yticks is not None:
            axes[i].set_yticks(yticks)
        
    
    # plt.ylim(-1,1)
    # plt.tight_layout() 
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=None)
    # plt.savefig(filename)
    return label_line


def find_ticks_and_labels(branches, shift=0.05):
    ticks = []
    labels = []
    LEN = len(branches)
    a_prev, b_prev = None, None
    for i in range(LEN):
        a,b = branches[i]['name'].split('-')
        if a is b_prev:
            print("a is b_prev")
            ticks.append([1])
            labels.append([r"${}$".format(b)])
            pass
        else:
            ticks.append([0, 1])
            labels.append([r"${}$".format(a), r"${}$".format(b)])
        a_prev, b_prev = a, b
        pass

    for i in range(LEN-1):
        if len(labels[i]) == 1 or len(labels[i+1]) == 1:
            continue
        print("comparing ", labels[i], " and ", labels[i+1])
        if labels[i][1] is labels[i+1][0]:
            print("label match")
        else:
            labels[i][1] = r"{}|".format(labels[i][1])
            ticks[i][1] -= shift
            print("label didn't match")

    for i in range(1, LEN):
        if len(labels[i]) == 1 or len(labels[i-1]) == 1:
            continue
        if "|" in labels[i-1][1]:
            ticks[i][0] += shift
            
    return ticks, labels




