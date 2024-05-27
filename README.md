# VeloxChemTutorial
Installation and startup
1)	Install miniconda. Instructions are found here: https://docs.anaconda.com/free/miniconda/miniconda-install/

2)	Download and store the veloxchem.yml file from this github repository (see list of files above)

3)	Open a terminal (mac or linux) or powershell (win). Go to the folder where the veloxchem.yml is located. Type:

conda env create -f veloxchem.yml

4)	When this has finished you have installed all code that you need to run VeloxChem and openMM on your computer. In conda you can have several environments, which makes it possible to use different versions of python codes without conflict. The environment you installed is called vlxenv. Before you can run code that is in that environment you need to type

conda activate vlxenv

in your terminal or powershell. 

Stop here for now. On the occation of the tutorial we will continue below.

5)	Next we will download out test notebook (MDconformations.ipynb), where we will run our first code. 

6)	In your terminal, go to the folder where the ipynb-file is located. If you have already activated the conda environment just type:

jupyter lab

7)  Since we will be working with some development version files we will need to copy them into the installation location (and overwrite the old ones with the same name. Download all .py files from here and copy the files atomtypeidentifier.py, forcefieldgenerator.py, and molecule.py into the python class folder. On my mac it is /opt/miniconda3/pkgs/veloxchem-1.0rc3-py311h80524ae_141/lib/python3.11/site-packages/veloxchem/ . To find it on a windows computer try to search for forcefieldgenerator.py .

8)  The file named openmmdynamics3.py should be copied into your working directory, the directory where you saved MDconformations.ipynb.

9)  Now it is time to open the MDconformations.ipynb notebook, where you will run you python code.

