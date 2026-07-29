# dfttools
DFT Tools. Extracting data from Quantum Espresso input/output files in similar format to that of materials project api. Analyzing/comparing


# Conda install 

```
conda create -n band python=3.10
conda activate band
conda install -c conda-forge pymatgen=2024.3.1 mp-api=0.43 emmet-core=0.84.2 scikit-image 
conda install -c conda-forge phonopy jupyter

```

```
conda install pytest
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


export PYTHONPATH=src

or
### pytest (pytest.ini must be in the root directory) 
```
pytest tests/test_basic.py
```


# Advanced tool for Wannier interpolation and integration of k-space integrals
# wannierberri

https://wannier-berri.org/index.html

conda install conda-forge::wannierberri irrep
conda install conda-forge::irrep

conda create -n berri
conda activate berri
conda install python=3.11 jupyter

pip3 install "wannierberri[all]"      


conda install conda-forge::wannierberri irrep ray-all pyfftw
conda install conda-forge::ray-all
conda install conda-forge::pyfftw






# Plotting fermi surface

conda install pyvista



# Fermisurfer

brew install imagemagick



# Fermisurfer
cd ~/Downloads/

wget https://github.com/mitsuaki1987/fermisurfer/archive/refs/tags/2.4.0.tar.gz

tar -xvf 2.4.0.tar.gz && cd fermisurfer-2.4.0/

## Install GTK3 in almalinux 10
sudo dnf install wxGTK-devel
sudo dnf install ImageMagick

sudo mkdir -p /opt/fermisurfer
sudo chown -R $USER:$USER /opt/fermisurfer
sudo chmod -R 755 /opt/fermisurfer


./configure --prefix=/opt/fermisurfer
make -j 32
make install

export PATH=/opt/fermisurfer/bin:$PATH

## Over SSH?
echo $DISPLAY



# 2026.06.28
# DFTtools with numpy 2.x
conda create -n band python=3.11
conda activate band
conda install -c conda-forge pymatgen mp-api emmet-core scikit-image pytest jupyter phonopy


# WannierBerri
conda create -n berri python=3.11
conda activate berri
conda install -c conda-forge ray-all pyfftw irrep
pip3 install "wannierberri[all]"



# Boltztrap env

```
conda create -n boltztrap python=3.11
conda activate boltztrap
conda install -c conda-forge jupyter mp-api pymatgen numpy scipy matplotlib cython spglib ase gfortran cmake make phonopy pyfftw vtk
conda install seaborn
```

On Linux, Open MPI is built with CUDA awareness but it is disabled by default.                  
To enable it, please set the environment variable                                               
OMPI_MCA_opal_cuda_support=true                                                                 
before launching your MPI processes.     



```
git clone https://gitlab.com/sousaw/BoltzTraP2.git
cd BoltzTraP2
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install .
```