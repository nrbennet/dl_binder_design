#!/usr/bin/env python3

'''
    Test that the ProteinMPNN environment is correctly installed
    and that this environment has access to the system's
    GPUs
'''

# PyRosetta
from pyrosetta import *
from rosetta import *
init()

# PyTorch
import torch

# Check for GPU access
if torch.cuda.is_available():
    print("GPU access is available")
    print('This environment passes all import tests')
else:
    print("GPU access is not available")
    print('For ProteinMPNN this is fine as it is very fast on CPU')
    print('This environment passes all import tests')
