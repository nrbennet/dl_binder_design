#!/bin/bash
rm check.point 2>/dev/null
../mpnn_fr/dl_interface_design.py -pdbdir inputs/pdbs -relax_cycles 0 -seqs_per_struct 4 -outpdbdir example3_out
