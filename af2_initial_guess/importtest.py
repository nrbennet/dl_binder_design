#!/usr/bin/env python

import os
import mock
import numpy as np
import sys
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

from include.silent_tools import silent_tools

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
