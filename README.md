# dfttools
DFT Tools. Extracting data from Quantum Espresso input/output files in similar format to that of materials project api. Analyzing/comparing


# Conda install 

```
conda create -n band python=3.10
conda activate band
conda install -c conda-forge pymatgen=2024.3.1 mp-api=0.43 emmet-core=0.84.2
```


# For developers
### Installing in editable mode
Run from root directory of the repository
```
pip install -e .
```

Editable mode means instead of copying your files into site-packages, it just adds a symbolic link back to your local `src/your_package` directory.

to uninstall
```
pip uninstall dfttools
```

or 
### Modify the pythonpath
```
export PYTHONPATH=./src:$PYTHONPATH
python tests/test_basic.py
```


