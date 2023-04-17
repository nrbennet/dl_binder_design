#!/usr/bin/env python

import os
import numpy as np
import sys
import datetime

from typing import Any, Mapping, Optional, Sequence, Tuple
import collections
from collections import OrderedDict

from timeit import default_timer as timer
import argparse

import io
#from Bio import PDB
#from Bio.PDB.Polypeptide import PPBuilder
#from Bio.PDB import PDBParser
#from Bio.PDB.mmcifio import MMCIFIO

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

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'include'))
from silent_tools import silent_tools

from pyrosetta import *
from rosetta import *
init( '-in:file:silent_struct_type binary' )

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-silent", dest="silent", required=True,
                        help="silent file to predict")
    parser.add_argument("-recycle", dest="recycle", type=int, default=3,
                        help="Number of recycle iterations to perform")

    args = parser.parse_args()
    return args

args = get_args()

model_name = "model_1_ptm"

model_config = config.model_config(model_name)
model_config.data.eval.num_ensemble = 1

model_config.data.common.num_recycle = args.recycle
model_config.model.num_recycle = args.recycle

model_config.model.embeddings_and_evoformer.initial_guess = True

model_config.data.common.max_extra_msa = 5
model_config.data.eval.max_msa_clusters = 5

# TODO change this dir
model_params = data.get_model_haiku_params(model_name=model_name, data_dir="/projects/ml/alphafold") # CHANGE THIS (directory where "params/ folder is")
model_runner = model.RunModel(model_config, model_params)

def get_seq_from_pdb( pdb_fn ):
  to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V' }

  seq = []
  seqstr = ''
  with open(pdb_fn) as fp:
    for line in fp:
      if line.startswith("TER"):
          seq.append(seqstr)
          seqstr = ''
      if not line.startswith("ATOM"):
        continue
      if line[12:16].strip() != "CA":
        continue
      resName = line[17:20]
      #
      seqstr += to1letter[resName]
  return seq

def af2_get_atom_positions( pdbfilename ) -> Tuple[np.ndarray, np.ndarray]:
  """Gets atom positions and mask from a list of Biopython Residues."""

  with open(pdbfilename, 'r') as pdb_file:
    lines = pdb_file.readlines()

  # indices of residues observed in the structure
  idx_s = [int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
  num_res = len(idx_s)

  all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
  all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                dtype=np.int64)

  residues = collections.defaultdict(list)
  # 4 BB + up to 10 SC atoms
  xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
  for l in lines:
    if l[:4] != "ATOM":
        continue
    resNo, atom, aa = int(l[22:26]), l[12:16], l[17:20]

    residues[ resNo ].append( ( atom.strip(), aa, [float(l[30:38]), float(l[38:46]), float(l[46:54])] ) )

  for resNo in residues:

    pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
    mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)

    for atom in residues[ resNo ]:
      atom_name = atom[0]
      x, y, z = atom[2]
      if atom_name in residue_constants.atom_order.keys():
        pos[residue_constants.atom_order[atom_name]] = [x, y, z]
        mask[residue_constants.atom_order[atom_name]] = 1.0
      elif atom_name.upper() == 'SE' and res.get_resname() == 'MSE':
        # Put the coordinates of the selenium atom in the sulphur column.
        pos[residue_constants.atom_order['SD']] = [x, y, z]
        mask[residue_constants.atom_order['SD']] = 1.0

    idx = idx_s.index(resNo) # This is the order they show up in the pdb
    all_positions[idx] = pos
    all_positions_mask[idx] = mask
  # _check_residue_distances(
  #     all_positions, all_positions_mask, max_ca_ca_distance) # AF2 checks this but if we want to allow massive truncations we don't want to check this

  return all_positions, all_positions_mask

def af2_all_atom_from_struct( pdbfilename, seq_list, just_target=False ):
  template_seq = ''.join( seq_list )

  # Parse a residue mask from the chainbreak sequence
  binder_len = len( seq_list[0] )
  residue_mask = [ int( i ) > binder_len for i in range( 1, len( template_seq ) + 1 ) ]

  all_atom_positions, all_atom_mask = af2_get_atom_positions( pdbfilename )

  all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])

  templates_all_atom_positions = []

  # Initially fill will all zero values
  for _ in template_seq:
    templates_all_atom_positions.append(
        jnp.zeros((residue_constants.atom_type_num, 3)))

  for idx, i in enumerate( template_seq ):
    if just_target and not residue_mask[ idx ]: continue

    templates_all_atom_positions[ idx ] = all_atom_positions[ idx ][0] # assign target indices to template coordinates

  return jnp.array(templates_all_atom_positions)

def template_from_struct( pdbfilename, seq_list ):

  template_seq = ''.join(seq_list)

  # Parse a residue mask from the chainbreak sequence
  binder_len = len( seq_list[0] )
  residue_mask = [ int( i ) > binder_len for i in range( 1, len( template_seq ) + 1 ) ]

  ret_all_atom_positions, ret_all_atom_mask = af2_get_atom_positions( pdbfilename )

  all_atom_positions = np.split(ret_all_atom_positions, ret_all_atom_positions.shape[0])
  all_atom_masks = np.split(ret_all_atom_mask, ret_all_atom_mask.shape[0])
  
  output_templates_sequence = []
  output_confidence_scores = []
  templates_all_atom_positions = []
  templates_all_atom_masks = []

  # Initially fill will all zero values
  for _ in template_seq:
    templates_all_atom_positions.append(
        np.zeros((residue_constants.atom_type_num, 3)))
    templates_all_atom_masks.append(np.zeros(residue_constants.atom_type_num))
    output_templates_sequence.append('-')
    output_confidence_scores.append(-1)
  
  confidence_scores = []
  for _ in template_seq: confidence_scores.append( 9 )

  for idx, i in enumerate( template_seq ):

    if not residue_mask[ idx ]: continue

    templates_all_atom_positions[ idx ] = all_atom_positions[ idx ][0] # assign target indices to template coordinates
    templates_all_atom_masks[ idx ] = all_atom_masks[ idx ][0]
    output_templates_sequence[ idx ] = template_seq[ idx ]
    output_confidence_scores[ idx ] = confidence_scores[ idx ] # 0-9 where higher is more confident

  output_templates_sequence = ''.join(output_templates_sequence)

  templates_aatype = residue_constants.sequence_to_onehot(
      output_templates_sequence, residue_constants.HHBLITS_AA_TO_ID)

  template_feat_dict = {'template_all_atom_positions': np.array(templates_all_atom_positions)[None],
       'template_all_atom_masks': np.array(templates_all_atom_masks)[None],
       'template_sequence': [output_templates_sequence.encode()],
       'template_aatype': np.array(templates_aatype)[None],
       'template_confidence_scores': np.array(output_confidence_scores)[None],
       'template_domain_names': ['none'.encode()],
       'template_release_date': ["none".encode()]}

  return template_feat_dict, ret_all_atom_positions, ret_all_atom_mask

def insert_chainbreaks( pose, binderlen ):

    conf = pose.conformation()
    conf.insert_chain_ending( binderlen )
    pose.set_new_conformation( conf )

    splits = pose.split_by_chain()
 
    newpose = splits[1]
    for i in range( 2, len( splits )+1 ):
        newpose.append_pose_by_jump( splits[i], newpose.size() )
 
    info = core.pose.PDBInfo( newpose, True )
    newpose.pdb_info( info )

    return newpose

    return final_dict

def get_final_dict(score_dict, string_dict):
    print(score_dict)
    final_dict = OrderedDict()
    keys_score = [] if score_dict is None else list(score_dict)
    keys_string = [] if string_dict is None else list(string_dict)

    all_keys = keys_score + keys_string

    argsort = sorted(range(len(all_keys)), key=lambda x: all_keys[x])

    for idx in argsort:
        key = all_keys[idx]

        if ( idx < len(keys_score) ):
            final_dict[key] = "%8.3f"%(score_dict[key])
        else:
            final_dict[key] = string_dict[key]

    return final_dict

def add2scorefile(tag, scorefilename, write_header=False, score_dict=None):
    with open(scorefilename, "a") as f:
        add_to_score_file_open(tag, f, write_header, score_dict)

def add_to_score_file_open(tag, f, write_header=False, score_dict=None, string_dict=None):
    final_dict = get_final_dict( score_dict, string_dict )
    if ( write_header ):
        f.write("SCORE:     %s description\n"%(" ".join(final_dict.keys())))
    scores_string = " ".join(final_dict.values())
    f.write("SCORE:     %s        %s\n"%(scores_string, tag))

def generate_scoredict( outtag, start_time, binderlen, prediction_result, scorefilename ):

  plddt_array = prediction_result['plddt']
  plddt = np.mean( plddt_array )
  plddt_binder = np.mean( plddt_array[:binderlen] )
  plddt_target = np.mean( plddt_array[binderlen:] )

  pae = prediction_result['predicted_aligned_error']
  pae_interaction1 = np.mean( pae[:binderlen,binderlen:] )
  pae_interaction2 = np.mean( pae[binderlen:,:binderlen] )
  pae_binder = np.mean( pae[:binderlen,:binderlen] )
  pae_target = np.mean( pae[binderlen:,binderlen:] )

  pae_interaction_total = ( pae_interaction1 + pae_interaction2 ) / 2

  time = timer() - start_time

  score_dict = {
          "plddt_total" : plddt,
          "plddt_binder" : plddt_binder,
          "plddt_target" : plddt_target,
          "pae_binder" : pae_binder,
          "pae_target" : pae_target,
          "pae_interaction" : pae_interaction_total,
          "time" : time
  }

  write_header=False
  if not os.path.isfile(scorefilename): write_header=True
  add2scorefile(outtag, scorefilename, write_header=write_header, score_dict=score_dict)
  
  print(score_dict)
  print( f"Tag: {outtag} reported success in {time} seconds" )

  return score_dict

def process_output( tag, binderlen, start, feature_dict, prediction_result, sfd_out, scorefilename ):
    structure_module = prediction_result['structure_module']
    this_protein = protein.Protein(
        aatype=feature_dict['aatype'][0],
           atom_positions=structure_module['final_atom_positions'][...],
           atom_mask=structure_module['final_atom_mask'][...],
           residue_index=feature_dict['residue_index'][0] + 1,
           b_factors=np.zeros_like(structure_module['final_atom_mask'][...]) )

    confidences = {}
    confidences['distogram'] = prediction_result['distogram']
    confidences['plddt'] = confidence.compute_plddt(
            prediction_result['predicted_lddt']['logits'][...])
    if 'predicted_aligned_error' in prediction_result:
        confidences.update(confidence.compute_predicted_aligned_error(
            prediction_result['predicted_aligned_error']['logits'][...],
            prediction_result['predicted_aligned_error']['breaks'][...]))
    
    unrelaxed_pdb_lines = protein.to_pdb(this_protein)
    
    outtag = f'{tag}_af2pred'
    unrelaxed_pdb_path = f'temp_af2output.pdb'
    with open(unrelaxed_pdb_path, 'w') as f: f.write(unrelaxed_pdb_lines)
    
    score_dict = generate_scoredict( outtag, start, binderlen, confidences, scorefilename )
    
    add2silent( outtag, unrelaxed_pdb_path, score_dict, binderlen, sfd_out )
    os.remove( unrelaxed_pdb_path )

def insert_truncations(residue_index, Ls):
    idx_res = residue_index
    for break_i in Ls:
        idx_res[break_i:] += 200
    residue_index = idx_res

    return residue_index

def add2silent( tag, pdb, score_dict, binderlen, sfd_out ):
    pose = pose_from_file( pdb )

    pose = insert_chainbreaks( pose, binderlen )

    struct = sfd_out.create_SilentStructOP()
    struct.fill_struct( pose, tag )

    for scorename, value in score_dict.items():
        struct.add_energy(scorename, value, 1)

    sfd_out.add_structure( struct )
    sfd_out.write_silent_struct( struct, "out.silent" )

def predict_structure(tag, feature_dict, binderlen, initial_guess, sfd_out, scorefilename, random_seed=0):  
    """Predicts structure using AlphaFold for the given sequence."""
    
    start = timer()
    print(f"running {model_name}")
    model_runner.params = model_params
    
    prediction_result = model_runner.apply( model_runner.params, jax.random.PRNGKey(0), feature_dict, initial_guess)
    
    process_output( tag, binderlen, start, feature_dict, prediction_result, sfd_out, scorefilename ) 
    
    print( f"Tag: {tag} reported success in {timer() - start} seconds" )

# Mostly taken from af2 source
# Used to detect truncation points
def check_residue_distances(all_positions,
                             all_positions_mask,
                             max_amide_distance):
    """Checks if the distance between unmasked neighbor residues is ok."""
    breaks = []
    
    c_position = residue_constants.atom_order['C']
    n_position = residue_constants.atom_order['N']
    prev_is_unmasked = False
    this_c = None
    for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):
        this_is_unmasked = bool(mask[c_position]) and bool(mask[n_position])
        if this_is_unmasked:
            this_n = coords[n_position]
            if prev_is_unmasked:
                distance = np.linalg.norm(this_n - prev_c)
                if distance > max_amide_distance:
                    breaks.append(i)
                    print( f'The distance between residues {i} and {i+1} is {distance:.2f} A' +
                        f' > limit {max_amide_distance} A.' )
                    print( f"I'm going to insert a chainbreak after residue {i}" )
            prev_c = coords[c_position]
        prev_is_unmasked = this_is_unmasked

    return breaks

def generate_feature_dict( pdbfile ):
  seq_list = get_seq_from_pdb(pdbfile)
  query_sequence = ''.join(seq_list)

  initial_guess = af2_all_atom_from_struct(pdbfile, seq_list, just_target=False)

  template_dict, all_atom_positions, all_atom_masks = template_from_struct(pdbfile, seq_list)
  
  # Gather features
  feature_dict = {
      **pipeline.make_sequence_features(sequence=query_sequence,
                                        description="none",
                                        num_res=len(query_sequence)),
      **pipeline.make_msa_features(msas=[[query_sequence]],
                                   deletion_matrices=[[[0]*len(query_sequence)]]),
      **template_dict
  }
  
  max_amide_distance = 3
  breaks = check_residue_distances(all_atom_positions, all_atom_masks, max_amide_distance)

  feature_dict['residue_index'] = insert_truncations(feature_dict['residue_index'], breaks)

  return feature_dict, initial_guess, len(seq_list[0]) 

def input_check( pdbfile, tag ):
    with open(pdbfile,'r') as f: lines = f.readlines()

    seen_indices = set()
    chain1 = True

    for line in lines:
        line = line.strip()
        
        if len(line) == 0: continue

        if line[:3] == "TER":
            chain1 = False
            continue

        if not line[:4] == "ATOM": continue

        if line[12:16].strip() == 'CA':
            # Only checking residue index at CA atom
            residx = line[22:27].strip()
            if residx in seen_indices:
                sys.exit( f"\nNon-unique residue indices detected for tag: {tag}. " +
                "This will cause AF2 to yield garbage outputs. Exiting." )

            seen_indices.add(residx)

        if ( not line[21:22].strip() == "A" ) and chain1:
            sys.exit( f"\nThe first chain in the pose must be the binder and it must be chain A. " +
                    f"Tag: {tag} does not satisfy this requirement. Exiting." )

def featurize(tag, sfd_in):
  
    # dump pdb
    pose = Pose()
    sfd_in.get_structure(tag).fill_pose(pose)
    pose.dump_pdb(tmppdb)
    
    # Input Checking
    # Must ensure that:
    # - All residue indices are unique
    # - The first chain is "A"
    
    input_check(tmppdb, tag)
    
    feature_dict, initial_guess, binderlen = generate_feature_dict(tmppdb)
    feature_dict = model_runner.process_features(feature_dict, random_seed=0)
    
    os.remove( tmppdb )

    return feature_dict, initial_guess, binderlen

# Checkpointing Functions

def record_checkpoint( tag, checkpoint_filename ):
    with open(checkpoint_filename, 'a') as f:
        f.write( tag )
        f.write( '\n' )

def determine_finished_structs( checkpoint_filename ):
    done_set = set()
    if not os.path.isfile( checkpoint_filename ): return done_set

    with open( checkpoint_filename, 'r' ) as f:
        for line in f:
            done_set.add( line.strip() )

    return done_set

# End Checkpointing Functions

################## Begin Main Function ##################

sfd_out = core.io.silent.SilentFileData( "out.silent", False, False, "binary", core.io.silent.SilentFileOptions())

sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
sfd_in.read_file(args.silent)

alltags = silent_tools.get_silent_index(args.silent)["tags"]
tmppdb = 'tmp.pdb'

checkpoint_filename = "check.point"
scorefilename = "out.sc"

finished_structs = determine_finished_structs( checkpoint_filename )

for idx,tag in enumerate(alltags):
  
  if tag in finished_structs:
    print( f"SKIPPING {tag}, since it was already run" )
    continue
  
  feature_dict, initial_guess, binderlen = featurize(tag, sfd_in)
  predict_structure(tag, feature_dict, binderlen, initial_guess, sfd_out, scorefilename)

  record_checkpoint( tag, checkpoint_filename )

print('done predicting')
