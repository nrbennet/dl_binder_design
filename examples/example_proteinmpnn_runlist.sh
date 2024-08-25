#!/bin/bash
rm check.point 2>/dev/null
python ../mpnn_fr/dl_interface_design.py -pdbdir inputs/pdbs -runlist inputs/tags.list -relax_cycles 1 -seqs_per_struct 1 -outpdbdir example2_out
