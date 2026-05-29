from dfttools import toCSV
from pathlib import Path

# DATA_PATH = Path("./tmp/Nb3S4_QE_data22.pkl")
# OUTPUT_DIR = Path("./tmp/")
# OUT_FILE = "Nb3S4"
# ENERGY_WINDOW = (-5.0, 5.0)
# toCSV.read_pkl_write_csv(DATA_PATH, OUT_FILE, ENERGY_WINDOW, OUTPUT_DIR)






# DATA_PATH = Path("./tmp/Nb3Se4_QE_data18.pkl")
# OUTPUT_DIR = Path("./tmp/")
# OUT_FILE = "Nb3Se4"
# ENERGY_WINDOW = (-5.0, 5.0)
# toCSV.read_pkl_write_csv(DATA_PATH, OUT_FILE, ENERGY_WINDOW, OUTPUT_DIR)





DATA_PATH = Path("./tmp/Nb3S4_PBEsol_NCPP_QE_data24.pkl")
OUTPUT_DIR = Path("./tmp/")
OUT_FILE = "Nb3S4_PBEsol_NCPP"
ENERGY_WINDOW = (-1.5, 1.5)
toCSV.read_pkl_write_csv(DATA_PATH, OUT_FILE, ENERGY_WINDOW, OUTPUT_DIR)