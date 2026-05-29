import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib as mpl
from dfttools import plot


mpl.rcParams['ytick.labelsize'] = 14



def bands_all_k(files):
    """
    plotting multiple files. 
    It's upto the user to make sure that all files are consistent.
    
    """
    first_plot = True
    colors = ['r-', 'g-', 'b-']
    shift = [0, -0.18, 0.115]
    shift = [0, -0.2, 0.15]
    Nlabels = ["N=-126", "N=-127", "N=-125"]

    for ifile, filename in enumerate(files):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            pass

        # For bands in range -0.4,0.4 eV range, we can skip few branches
        branches = data['branches']

        branches = branches[0], branches[1], branches[2], branches[3], branches[7], branches[8]

        ticks, labels = plot.find_ticks_and_labels(branches)
        # ticks[5][0] += 0.01
        # ticks[5][0] += 0.01
        
        ticks[4][0] += 0.01
        ticks[5][0] += 0.04
        ticks[5][1] -= 0.12
        labels[5][1] = "{}|".format(labels[5][1])


        if first_plot:
            width_ratios = list(map(lambda x: x['end_index']-x['start_index'], branches))
            width_ratios.append(np.max(width_ratios))
            # print(width_ratios)
            fig, axes = plt.subplots(1, len(branches)+1, figsize=(10, 5), 
                sharey=True, gridspec_kw={"width_ratios": width_ratios}, dpi=200)
            first_plot = False
            pass


        plot.plot_bands_v3(data, axesin=axes, fermi_shift=shift[ifile], symbol_=colors[ifile], branches=branches)
        axes[0].set_ylabel(r"$E-E_F (eV)$", fontsize=20)
        axes[0].set_ylabel(r"$E-E_F (eV)-\Delta$", fontsize=20)

        # branches = data['branches']
        # for i, branch in enumerate(branches):
        #     if i > 0:
        #         prev_branch = branches[i-1]
        #         prev_end_index = prev_branch['end_index']
        #         if branch['start_index'] == prev_end_index:
        #             axes[i].set_xlabel(r"${}$".format(branch['name']), fontsize=14)
        #         # else:
        #         #     axes[i].set_xlabel(r"${}$".format(branch['name']), fontsize=14)
        #     else:
        #         axes[i].set_xlabel(r"${}$".format(branch['name']), fontsize=14)



        from matplotlib.ticker import AutoMinorLocator
        from matplotlib.ticker import MultipleLocator
        for i in range(len(branches)):
            axes[i].axhline(0, 0, 1, color='k', linestyle='--')
            axes[i].set_xticks(ticks[i])
            axes[i].set_xticklabels(labels[i])
            axes[i].yaxis.set_minor_locator(MultipleLocator(0.25))
            axes[i].tick_params(axis='y', labelsize=18)
            axes[i].tick_params(axis='x', labelsize=18)
            # axes[i].set_yticks([-1, -0.5, 0, 0.5, 1])
            # axes[i].set_yticks([-0.4, -0.2, 0,  0.2, 0.4, 0.6])
            axes[i].tick_params(axis='y', which='major', direction='in', length=8, width=1.5)
            axes[i].tick_params(axis='y', which='minor', direction='in', length=4)

        axes[3].set_xlabel("wave vector", fontsize=18)

        # Turn on minor ticks


        for ax in axes[1:]:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.tick_params(axis='x', which='both', bottom=False)
                
        # plt.title(fig_title)
        plt.ylim((-0.3, 0.13))
        



        ax = axes[-1]
        x, y = data['dos'].T
        print(x.shape)
        print(y.shape)
        ax.plot(y, x-data['e_fermi']-shift[ifile], colors[ifile], label=r"$E_F={:.2f},{},\Delta={}$".format(data['e_fermi'],Nlabels[ifile], shift[ifile]))
        ax.axhline(0, 0, 1, color='k', linestyle='--')
        ax.set_xlim(0,28)
        ax.set_xlabel(r"DOS (arb. unit)", fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.set_xticks([0, 20])
        ax.tick_params(axis='x', which='both', bottom=True)
        ax.legend()
        
        # ax.ylim(0, 25)
        pass


    plt.tight_layout() 
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=None)
    plt.savefig("Nb3S4_PBEsol_NCPP_QE_data-all-k-zoom1-compare.png")



def bands_kz(files):
    first_plot = True
    for filename in files:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            pass

        # For bands in range -0.4,0.4 eV range, we can skip few branches
        branches = data['branches']

        # branches = branches[0], branches[2], branches[1]

        ticks, labels = plot.find_ticks_and_labels(branches)
        ticks[0][1] -= 0.04
        ticks[1][0] += 0.07
        ticks[1][1] -= 0.04
        ticks[2][0] += 0.07
        ticks[2][1] -= 0.17
        labels[2][1] = "{}|".format(labels[2][1])


        if first_plot:
            width_ratios = list(map(lambda x: x['end_index']-x['start_index'], branches))
            width_ratios.append(np.max(width_ratios)-35)
            # print(width_ratios)
            fig, axes = plt.subplots(1, len(branches)+1, figsize=(6, 5), 
                sharey=True, gridspec_kw={"width_ratios": width_ratios}, dpi=200)
            first_plot = False
            pass


        plot.plot_bands_v3(data, axesin=axes, fermi_factor=1, symbol_='r-', branches=branches)
        axes[0].set_ylabel(r"$E-E_F (eV)$", fontsize=20)

        # branches = data['branches']
        # for i, branch in enumerate(branches):
        #     if i > 0:
        #         prev_branch = branches[i-1]
        #         prev_end_index = prev_branch['end_index']
        #         if branch['start_index'] == prev_end_index:
        #             axes[i].set_xlabel(r"${}$".format(branch['name']), fontsize=14)
        #         # else:
        #         #     axes[i].set_xlabel(r"${}$".format(branch['name']), fontsize=14)
        #     else:
        #         axes[i].set_xlabel(r"${}$".format(branch['name']), fontsize=14)



        from matplotlib.ticker import AutoMinorLocator
        from matplotlib.ticker import MultipleLocator
        for i in range(len(branches)):
            axes[i].axhline(0, 0, 1, color='k', linestyle='--')
            axes[i].set_xticks(ticks[i])
            axes[i].set_xticklabels(labels[i])
            axes[i].yaxis.set_minor_locator(MultipleLocator(0.25))
            axes[i].tick_params(axis='y', labelsize=18)
            axes[i].tick_params(axis='x', labelsize=18)
            # axes[i].set_yticks([-1, -0.5, 0, 0.5, 1])
            # axes[i].set_yticks([-0.4, -0.2, 0,  0.2, 0.4, 0.6])
            axes[i].tick_params(axis='y', which='major', direction='in', length=8, width=1.5)
            axes[i].tick_params(axis='y', which='minor', direction='in', length=4)

        axes[1].set_xlabel("wave vector", fontsize=18)

        # Turn on minor ticks


        for ax in axes[1:]:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        for ax in axes[:-1]:
            ax.tick_params(axis='x', which='both', bottom=False)
                    
        # plt.title(fig_title)
        plt.ylim((-0.3, 0.3))



        ax = axes[-1]
        x, y = data['dos'].T
        ax.plot(y, x-data['e_fermi'])
        ax.axhline(0, 0, 1, color='k', linestyle='--')
        ax.set_xlim(0,14)
        ax.set_xlabel("DOS (arb. unit)", fontsize=10)
        ax.tick_params(axis='x', labelsize=18)
        ax.set_xticks([0, 10])
        # ax.tick_params(axis='x', which='both', bottom=True)

        # ax.ylim(0, 25)



        plt.tight_layout() 
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=None)
        pass

    plt.savefig("Nb3S4_PBEsol_NCPP_QE_data-kz-zoom1-compare.png")



def bands_kx(files):
    first_plot = True
    colors = ['r-', 'b-']
    for i, filename in enumerate(files):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            pass

        # For bands in range -0.4,0.4 eV range, we can skip few branches
        branches = data['branches']

        branches = branches[0], branches[1], branches[2]

        ticks, labels = plot.find_ticks_and_labels(branches)
        # ticks[5][0] += 0.01
        ticks[2][0] -= 0.12
        labels[2][0] = "{}|".format(labels[2][0])


        if first_plot:
            width_ratios = list(map(lambda x: x['end_index']-x['start_index'], branches))
            width_ratios.append(np.max(width_ratios))
            # print(width_ratios)
            fig, axes = plt.subplots(1, len(branches)+1, figsize=(6, 5), 
                sharey=True, gridspec_kw={"width_ratios": width_ratios}, dpi=200)
            first_plot = False
            pass


        plot.plot_bands_v3(data, axesin=axes, fermi_factor=1, symbol_=colors[i], branches=branches)
        axes[0].set_ylabel(r"$E-E_F (eV)$", fontsize=20)

        # branches = data['branches']
        # for i, branch in enumerate(branches):
        #     if i > 0:
        #         prev_branch = branches[i-1]
        #         prev_end_index = prev_branch['end_index']
        #         if branch['start_index'] == prev_end_index:
        #             axes[i].set_xlabel(r"${}$".format(branch['name']), fontsize=14)
        #         # else:
        #         #     axes[i].set_xlabel(r"${}$".format(branch['name']), fontsize=14)
        #     else:
        #         axes[i].set_xlabel(r"${}$".format(branch['name']), fontsize=14)



        from matplotlib.ticker import AutoMinorLocator
        from matplotlib.ticker import MultipleLocator
        for i in range(len(branches)):
            axes[i].axhline(0, 0, 1, color='k', linestyle='--')
            axes[i].set_xticks(ticks[i])
            axes[i].set_xticklabels(labels[i])
            axes[i].yaxis.set_minor_locator(MultipleLocator(0.25))
            axes[i].tick_params(axis='y', labelsize=18)
            axes[i].tick_params(axis='x', labelsize=18)
            # axes[i].set_yticks([-1, -0.5, 0, 0.5, 1])
            # axes[i].set_yticks([-0.4, -0.2, 0,  0.2, 0.4, 0.6])
            axes[i].tick_params(axis='y', which='major', direction='in', length=8, width=1.5)
            axes[i].tick_params(axis='y', which='minor', direction='in', length=4)

        axes[1].set_xlabel("wave vector", fontsize=18)

        # Turn on minor ticks


        for ax in axes[1:]:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.tick_params(axis='x', which='both', bottom=False)
                
        # plt.title(fig_title)
        plt.ylim((-0.3, 0.3))


        
        ax = axes[-1]
        x, y = data['dos'].T
        ax.plot(y, x-data['e_fermi'])
        ax.axhline(0, 0, 1, color='k', linestyle='--')
        ax.set_xlim(0,35)
        ax.tick_params(axis='x', which='both', bottom=True)
        ax.set_xlabel("DOS (arb. unit)", fontsize=16)
        ax.tick_params(axis='x', labelsize=18)
        ax.set_xticks([0, 30])
        ax.tick_params(axis='x', which='both', bottom=True)
        
        # ax.ylim(0, 25)



        plt.tight_layout() 
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=None)
        pass

    plt.savefig("Nb3S4_PBEsol_NCPP_QE_data-kx-zoom1-compare.png")



if __name__ == "__main__":
    files = [
        "./tcl9-PBEsol/Nb3S4_PBEsol_NCPP_QE_data24.pkl",
        "./tcl10-PBEsol/run2/Nb3S4_PBEsol_NCPP_QE_data24.pkl",
        "./tcl10-PBEsol/run3/Nb3S4_PBEsol_NCPP_QE_data24.pkl"
             ]
    bands_all_k(files)
    # bands_kx()
    # bands_kz()

