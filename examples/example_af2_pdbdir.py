#!/bin/bash
rm check.point 2>/dev/null
../af2_initial_guess/predict.py -pdbdir inputs/proteinmpnn_output_pdbs -outpdbdir out_example5
