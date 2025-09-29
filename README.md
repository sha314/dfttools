# dfttools
DFT Tools. Extracting data from Quantum Espresso input/output files in similar format to that of materials project api. Analyzing/comparing



# For developers
### Installing in editable mode
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


