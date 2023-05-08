#!/usr/bin/env python

import os, sys
from collections import OrderedDict
import time
import argparse
import time
import torch

import util_protein_mpnn as mpnn_util

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'include'))
from silent_tools import silent_tools

from pyrosetta import *
from pyrosetta.rosetta import *
init("-beta_nov16 -in:file:silent_struct_type binary")

#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()
parser.add_argument( "-silent", type=str, required=True, help='The name of a silent file to run this metric on. pdbs are not accepted at this point in time' )
parser.add_argument( "-checkpoint_path", type=str, required=True, help='The path to the set of ProteinMPNN model weights that you would like to use' )
parser.add_argument( "-relax_cycles", type=int, default="1", help="The number of ProteinMPNN->FastRelax cycles to perform (default 2)" )
parser.add_argument( "-output_intermediates", action="store_true", help='Whether to write all intermediate sequences from the relax cycles to disc (defaut False)' )
parser.add_argument( "-temperature", type=float, default=0.000001, help='The temperature to use for ProteinMPNN sampling (default 0)' )
parser.add_argument( "-augment_eps", type=float, default=0, help='The variance of random noise to add to the atomic coordinates (default 0)' )
parser.add_argument( "-omit_AAs", type=str, default='X', help='A string off all residue types (one letter case-insensitive) that you would not like to use for design. Letters not corresponding to residue types will be ignored' )
parser.add_argument( "-num_connections", type=int, default=48, help='Number of neighbors each residue is connected to (default 48)' )
parser.add_argument( "-fix_FIXED_res", action="store_true", help='Whether to fix the sequence of residues labelled as FIXED or not (default False)' )

args = parser.parse_args( sys.argv[1:] )
silent = args.__getattribute__("silent")

omit_AAs = [ letter for letter in args.omit_AAs.upper() if letter in list("ARNDCQEGHILKMFPSTWYVX") ]

rundir = os.path.dirname(os.path.realpath(__file__))

xml = rundir + "/RosettaFastRelaxUtil.xml"
objs = protocols.rosetta_scripts.XmlObjects.create_from_file( xml )

# Load the movers we will need

FastRelax = objs.get_mover( 'FastRelax' )

silent_out = "out.silent"

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
states = len(alpha_1)
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_N = {a:n for n,a in enumerate(alpha_1)}
aa_3_N = {a:n for n,a in enumerate(alpha_3)}
aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

# I/O Functions

def add2silent( pose, tag, sfd_out ):
    struct = sfd_out.create_SilentStructOP()
    struct.fill_struct( pose, tag )
    sfd_out.add_structure( struct )
    sfd_out.write_silent_struct( struct, "out.silent" )

# End I/O Functions

def my_rstrip(string, strip):
    if (string.endswith(strip)):
        return string[:-len(strip)]
    return string

def thread_mpnn_seq( pose, binder_seq ):
    rsd_set = pose.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )

    for resi, mut_to in enumerate( binder_seq ):
        resi += 1 # 1 indexing
        name3 = aa_1_3[ mut_to ]
        new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )
        pose.replace_residue( resi, new_res, True )
    
    return pose

def sequence_optimize( pdbfile, chains, model, fixed_positions_dict ):
    
    t0 = time.time()

    feature_dict = mpnn_util.generate_seqopt_features( pdbfile, chains )

    seq_per_struct = 1
    arg_dict = mpnn_util.set_default_args( seq_per_struct, omit_AAs=omit_AAs )
    arg_dict['temperature'] = args.temperature

    masked_chains = [ chains[0] ]
    visible_chains = [ chains[1] ]

    sequences = mpnn_util.generate_sequences( model, device, feature_dict, arg_dict, masked_chains, visible_chains, fixed_positions_dict )
    
    print( f"MPNN generated {len(sequences)} sequences in {int( time.time() - t0 )} seconds" ) 

    return sequences

def add2silent( pose, tag, sfd_out ):
    struct = sfd_out.create_SilentStructOP()
    struct.fill_struct( pose, tag )
    sfd_out.add_structure( struct )
    sfd_out.write_silent_struct( struct, "out.silent" )

def get_chains( pose ):
    lengths = [ p.size() for p in pose.split_by_chain() ]
    endA = pose.split_by_chain()[1].size()
    endB = endA + pose.split_by_chain()[2].size()

    chains = [ pose.pdb_info().chain( i ) for i in [ endA, endB ] ]

    return chains

def get_fixed_res(pose):
    fixed_res = []
    pdb_info = pose.pdb_info()
    endA = pose.split_by_chain()[1].size()
    for i in range(1,endA+1):
        if pdb_info.res_haslabel(i,"FIXED"):
            fixed_res.append(i)
    return fixed_res

def relax_pose( pose ):
    FastRelax.apply( pose )
    return pose

def get_fixed_res(pose):
    fixed_res = []
    pdb_info = pose.pdb_info()
    endA = pose.split_by_chain()[1].size()
    for i in range(1,endA+1):
        if pdb_info.res_haslabel(i,"FIXED"):
            fixed_res.append(i)
    return fixed_res

def dl_design( pose, tag, og_struct, mpnn_model, sfd_out ):

    tot_t0 = time.time()

    prefix = f"{tag}_dldesign"
    pdbfile = f"tmp.pdb"

    if args.fix_FIXED_res:
        fixed_res = get_fixed_res( pose )
    else:
        fixed_res = []

    fixed_positions_dict = None
    if len(fixed_res)>0:
        fixed_positions_dict = {}
        fixed_positions_dict[my_rstrip(pdbfile,'.pdb')] = {"A":fixed_res,"B":[]}
        print("Found residues with FIXED label, fixing the following residues: ", fixed_positions_dict[my_rstrip(pdbfile,'.pdb')])

    for cycle in range(args.relax_cycles):
        pose.dump_pdb( pdbfile )
        chains = get_chains( pose )

        seqs_scores = sequence_optimize( pdbfile, chains, mpnn_model, fixed_positions_dict )
        os.remove( pdbfile )

        seq, mpnn_score = seqs_scores[0] # We know there is only one entry
        pose = thread_mpnn_seq( pose, seq )

        pose = relax_pose(pose)

        if args.output_intermediates:
            tag = f"{prefix}_0_cycle{cycle}"
            add2silent( pose, tag, sfd_out )

    # Do the final sequence assignment
    pose.dump_pdb( pdbfile )
    chains = get_chains( pose )

    seqs_scores = sequence_optimize( pdbfile, chains, mpnn_model, fixed_positions_dict )
    os.remove( pdbfile )

    seq, mpnn_score = seqs_scores[0] # We know there is only one entry
    pose = thread_mpnn_seq( pose, seq )

    tag = f"{prefix}_0_cycle{cycle}"

    add2silent( pose, tag, sfd_out )

def main( pdb, silent_structure, mpnn_model, sfd_in, sfd_out ):

    t0 = time.time()
    print( "Attempting pose: %s"%pdb )
    
    pose = Pose()
    sfd_in.get_structure( pdb ).fill_pose( pose )

    dl_design( pose, pdb, silent_structure, mpnn_model, sfd_out )

    seconds = int(time.time() - t0)

    print( f"{pdb} reported success. 1 designs generated in {seconds} seconds" )

# Checkpointing Functions

def record_checkpoint( pdb, checkpoint_filename ):
    with open( checkpoint_filename, 'a' ) as f:
        f.write( pdb )
        f.write( '\n' )

def determine_finished_structs( checkpoint_filename ):
    done_set = set()
    if not os.path.isfile( checkpoint_filename ): return done_set

    with open( checkpoint_filename, 'r' ) as f:
        for line in f:
            done_set.add( line.strip() )

    return done_set

# End Checkpointing Functions

#################################
# Begin Main Loop
#################################

silent_index = silent_tools.get_silent_index(silent)

sfd_out = core.io.silent.SilentFileData("out.silent", False, False, "binary", core.io.silent.SilentFileOptions())

sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
sfd_in.read_file(silent)

sf_open = open(silent)

checkpoint_filename = "check.point"
debug = True

if torch.cuda.is_available():
    print('Found GPU will run MPNN on GPU')
    device = "cuda:0"
else:
    print('No GPU found, running MPNN on CPU')
    device = "cpu"

finished_structs = determine_finished_structs( checkpoint_filename )
mpnn_model = mpnn_util.init_seq_optimize_model(device, hidden_dim=128, num_layers=3, backbone_noise=args.augment_eps, num_connections=args.num_connections, checkpoint_path=args.checkpoint_path)

for pdb in silent_index['tags']:

    if pdb in finished_structs: continue

    silent_structure = silent_tools.get_silent_structure_file_open( sf_open, silent_index, pdb )

    main( pdb, silent_structure, mpnn_model, sfd_in, sfd_out )

    record_checkpoint( pdb, checkpoint_filename )



