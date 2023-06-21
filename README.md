# dl_binder_design

This repo contains the scripts described in the paper [Improving de novo Protein Binder Design with Deep Learning](https://www.nature.com/articles/s41467-023-38328-5).

![forgithub](https://github.com/nrbennet/dl_binder_design/assets/56419265/4c5d6a05-d2fb-4c7b-b1e0-0c743b2114eb)


## Setup:

## Conda Environment
- Ensure that you have the Anaconda or Miniconda package manager
- Ensure that you have the PyRosetta channel included in your `~/.condarc`
- Your `~/.condarc` should look something like this:
```
channels: 
- https://USERNAME:PASSWORD@conda.graylab.jhu.edu
- conda-forge
- defaults
```
- More information about conda installing PyRosetta may be found here: https://www.pyrosetta.org/downloads
- Clone this repo
- Navigate to <base_dir>/include
- Run `conda env create -f dl_binder_design.yml`

## Clone ProteinMPNN
This repo requires the code from ProteinMPNN to work. It expects this code to be in the mpnn_fr directory so we can just clone it to be there
- Naviate to <base_dir>/mpnn_fr
- Run `git clone https://github.com/dauparas/ProteinMPNN.git`

## Silent Tools
The scripts contained in this repository work with a type of file called silent files. These are essentially a bunch of compressed .pdb files that are all stored in one file. Working with silent files is conventient and saves a lot of disk space when dealing with many thousands of structures.

Brian Coventry wrote a bunch of really nice commandline tools (called silent_tools) to manipulate silent files. These tools are included in this repository but may also be downloaded separately from [this](https://github.com/bcov77/silent_tools) GitHub repo.

The two commands that allow you to go from pdb to silent file and back are the following:

pdbs to silent: `<base_dir>/silentfrompdbs *.pdb > my_designs.silent` 

silent to pdbs: `<base_dir>/silentextract all_structs.silent`

NOTE: Some silent tools require PyRosetta and will fail if run in a Python environment without access to PyRosetta.

## Ensure Access to AlphaFold2 model weights
The scripts in this repository expect AF2 weights to be in <base_dir>/model_weights/params and will fail to run if the weights are not there. If you already have AF2 params_model_1_ptm.npz weights downloaded then you may simply copy them to <base_dir>/model_weights/params or create a symlink. If you do not have these weights downloaded you will have to download them from DeepMind's repository, this can be done as follows:

cd <base_dir>/af2_initial_guess/model_weights/params
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar --extract --verbose --file=alphafold_params_2022-12-06.tar 

## AlphaFold2 Complex Prediction

Running the interface prediction script is simple:

```
<base_dir>/af2_initial_guess/predict.py -silent my_designs.silent
```

This will create a file titled `out.silent` containing the AF2 predictions of your designs. It will also output a file titled `out.sc` with the scores of the designs, `pae_interaction` is the score that showed the most predictivity in the experiments performed in the paper.

With the refactor, this script is now able to read and write PDB and silent files as well as perform both monomer and complex predictions. The arguments for these can be listed by running:

```
<base_dir>/af2_initial_guess/predict.py -h
```

NOTE: This script expects your binder design to be the first chain it receives. The binder will be predicted from single sequence and with an intial guess. The target chains will be fixed to the input structure. The script also expects your residue indices to be unique, ie. your binder and target cannot both start with residue 1.

## ProteinMPNN-FastRelax Binder Design

Running design with ProteinMPNN-FastRelax cycling is also simple:

```
<base_dir>/mpnn_fr/dl_interface_design.py -silent my_designs.silent
```

This will create a file titled `out.silent` containing your designs. This file can be fed directly to AF2 interface prediction.

With the refactor, this script is now able to read and write both PDB files and silent files. I have also added more informative argument messages which can be accessed by running:

```
<base_dir>/mpnn_fr/dl_interface_design.py -h
```

NOTE: This script expects your binder design to be the first chain it receives. This script is robust to non-unique indices, unlike the AF2 interface script.

### Running ProteinMPNN with Fixed Residues

If you used RFdiffusion to generate your binder designs and would like to fix a region, you can use the following command to add 'FIXED' labels to your pdbs which will be recognized by the ProteinMPNN scripts. Thanks to Preetham Venkatesh for writing this!

```
python <base_dir>/helper_scripts/addFIXEDlabels.py --pdbdir /dir/of/pdbs --trbdir /dir/of/trbs --verbose
```

These pdb files can be collected into a silent file (or just used as PDB files) and run through the ProteinMPNN script which will detect the FIXED labels and keep those sequence positions fixed.

