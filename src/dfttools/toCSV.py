import pickle
import csv
from pathlib import Path

import numpy as np





# ==============================
# Helper Functions
# ==============================
def load_data(pkl_path):
    """Load pickled band structure data."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def filter_bands(bands, e_fermi, energy_range):
    """Shift bands by Fermi energy and filter within energy window."""
    low, high = energy_range

    shifted = bands - e_fermi
    mask = (shifted >= low) & (shifted <= high)

    # Keep rows (k-points) with at least one energy in range
    rows_keep = np.any(mask, axis=1)

    # Keep bands with at least one energy in range
    cols_keep = np.any(mask, axis=0)

    return shifted[rows_keep][:, cols_keep]


def save_bands_to_csv(bands, filename):
    """Save filtered bands to CSV (bands as columns)."""
    header = ",".join([f"E{i+1}" for i in range(bands.shape[0])])
    np.savetxt(filename, bands.T, delimiter=",", header=header, comments="#")


def save_branches_to_csv(branches, filename):
    """Save branch information to CSV."""
    fieldnames = ["name", "start_index", "end_index"]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(branches)



def read_pkl_write_csv(DATA_PATH, OUT_FILE, ENERGY_WINDOW=(-1.0, 1.0), OUTPUT_DIR="."):
    data = load_data(DATA_PATH)

    bands = data["bands"]
    kpoints = data["kpoints"]
    branches = data["branches"]
    e_fermi = data["e_fermi"]

    print("Data keys:", data.keys())
    print("Bands shape:", bands.shape)
    print("Kpoints shape:", kpoints.shape)
    print("Number of branches:", len(branches))

    filtered_bands = filter_bands(bands, e_fermi, ENERGY_WINDOW)
    print("Filtered bands shape:", filtered_bands.shape)

    OUTPUT_DIR.mkdir(exist_ok=True)

    save_bands_to_csv(filtered_bands, OUTPUT_DIR / f"{OUT_FILE}_bands.csv")
    save_branches_to_csv(branches, OUTPUT_DIR / f"{OUT_FILE}_branches.csv")

    print("Export complete.")


if __name__ == "__main__":
    # ==============================
    # Configuration
    # ==============================
    DATA_PATH = Path("./Nb3S4_QE_data22.pkl")
    OUTPUT_DIR = Path(".")
    OUT_FILE = "Nb3S4"
    ENERGY_WINDOW = (-1.0, 1.0)
    read_pkl_write_csv(DATA_PATH, OUT_FILE, ENERGY_WINDOW, OUTPUT_DIR)
    