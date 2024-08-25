#!/bin/bash
rm check.point 2>/dev/null
../mpnn_fr/dl_interface_design.py -silent inputs/in.silent -relax_cycles 0 -seqs_per_struct 4 -outsilent example1_out.silent
