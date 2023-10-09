#!/bin/bash
rm check.point 2>/dev/null
../af2_initial_guess/predict.py -silent inputs/proteinmpnn_output.silent -outsilent out_example4.silent
