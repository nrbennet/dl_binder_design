#!/usr/bin/env python

'''
    Test the imports required by the AF2 initial guess scripts

    This will also test whether the currently installed version of JAX is able to 
    run on the GPU.

    Run this on a gpu node like this:
    python importtest.py
'''

import os
import mock
import numpy as np
import sys
# Get the path of the script
scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{scriptdir}/../../af2_initial_guess')

import datetime

from typing import Any, Mapping, Optional, Sequence, Tuple
import collections
from collections import OrderedDict

from timeit import default_timer as timer
import argparse

import io
from Bio import PDB
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import PDBParser
from Bio.PDB.mmcifio import MMCIFIO

import scipy
import jax
import jax.numpy as jnp

from alphafold.common import residue_constants
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.data import mmcif_parsing
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data.tools import hhsearch

sys.path.append(f'{scriptdir}/../..')
from include.silent_tools import silent_tools

# PyRosetta install test
print("/"*200)
print("Testing PyRosetta install. If this script errors before you see a PyRosetta success message then you " + \
      "have an issue with your PyRosetta install")
print("/"*200)

from pyrosetta import *
from rosetta import *
init()

print("/"*70)
print("PyRosetta installation was successful!")
print("/"*70)

print("\n")

from jax.lib import xla_bridge
device = xla_bridge.get_backend().platform

if device == 'gpu':
    print('/'*70)
    print('Found a GPU! This environment passes all import tests')
    print('/'*70)
else:
    print('/'*70)
    print('No GPU found! This environment passes all import tests, but will not be able to use a GPU!!')
    print('/'*70)
