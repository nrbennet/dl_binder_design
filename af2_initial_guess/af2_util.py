#!/usr/bin/env python3

import numpy as np
from typing import Tuple
import collections
from collections import OrderedDict
import os

from alphafold.common import residue_constants

import jax.numpy as jnp

from pyrosetta import *
from rosetta import *
# I don't think we need to call init() here but I'm not sure

def get_seq_from_pdb( pdb_fn ) -> str:
    '''
    Given a pdb file, return the sequence of the protein as a string.
    '''

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
            
            seqstr += to1letter[resName]

    return seq

def generate_template_features(
                                seq: str,
                                all_atom_positions: np.ndarray,
                                all_atom_masks: np.ndarray,
                                residue_mask: list
                                ) -> dict:
    '''
    Given the sequence and all atom positions and masks, generate the template features.
    Residues which are False in the residue mask are not included in the template features,
    this means they will be free to be predicted by the model.
    '''

    # Split the all atom positions and masks into a list of arrays for easier manipulation
    all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])
    all_atom_masks = np.split(all_atom_masks, all_atom_masks.shape[0])

    output_templates_sequence = []
    output_confidence_scores = []
    templates_all_atom_positions = []
    templates_all_atom_masks = []

    # Initially fill will all zero values
    for _ in seq:
        templates_all_atom_positions.append(
            np.zeros((residue_constants.atom_type_num, 3)))
        templates_all_atom_masks.append(np.zeros(residue_constants.atom_type_num))
        output_templates_sequence.append('-')
        output_confidence_scores.append(-1)

    confidence_scores = []
    for _ in seq: confidence_scores.append( 9 )

    for idx, i in enumerate(seq):

        if not residue_mask[ idx ]: continue

        templates_all_atom_positions[ idx ] = all_atom_positions[ idx ][0] # assign target indices to template coordinates
        templates_all_atom_masks[ idx ] = all_atom_masks[ idx ][0]
        output_templates_sequence[ idx ] = seq[ idx ]
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

    return template_feat_dict    

def parse_initial_guess(all_atom_positions) -> jnp.ndarray:
    '''
    Given a numpy array of all atom positions, return a jax array of the initial guess
    '''

    list_all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])

    templates_all_atom_positions = []

    # Initially fill with zeros
    for _ in list_all_atom_positions:
        templates_all_atom_positions.append(jnp.zeros((residue_constants.atom_type_num, 3)))

    for idx in range(len(list_all_atom_positions)):
        templates_all_atom_positions[idx] = list_all_atom_positions[idx][0] 

    return jnp.array(templates_all_atom_positions)

def af2_get_atom_positions(pose, tmp_fn) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Given a pose, return the AF2 atom positions array and atom mask array for the protein.
    '''

    # write pose to pdb file
    pose.dump_pdb(tmp_fn)

    with open(tmp_fn, 'r') as pdb_file:
        lines = pdb_file.readlines()

    # Delete the temporary file
    os.remove(tmp_fn)

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

    return all_positions, all_positions_mask

def insert_truncations(residue_index, Ls) -> np.ndarray:
    '''
    Given the residue index feature and the absolute indices of the truncations,
    insert the truncations into the residue index feature.

    Args:
        residue_index (np.ndarray) : [L] The residue index feature.

        Ls (list)                  : The absolute indices of the chainbreaks.
                                     Chainbreaks will be inserted after these zero-indexed indices.
    '''

    idx_res = residue_index
    for break_i in Ls:
        idx_res[break_i:] += 200
    
    residue_index = idx_res

    return residue_index

def get_final_dict(score_dict, string_dict) -> OrderedDict:
    '''
    Given dictionaries of numerical scores and a string scores, return a sorted dictionary
    of the scores, ready to be written to the scorefile.
    '''

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

def add2scorefile(tag, scorefilename, write_header=False, score_dict=None, string_dict=None) -> None:
    '''
    Given a score filename, add scores to the scorefile.

    Args:
        tag (str) : The tag to add to the scorefile.

        scorefilename (str) : The score filename to add the scores to.

        write_header (bool) : Whether to write the header or not.
                              The first tag written to the scorefile should have this set to True.

        score_dict (dict) : The dictionary of numerical scores to add to the scorefile.

        string_dict (dict) : The dictionary of string scores to add to the scorefile.
    '''

    with open(scorefilename, "a") as f:
        final_dict = get_final_dict( score_dict, string_dict )

        if ( write_header ):
            f.write("SCORE:     %s description\n"%(" ".join(final_dict.keys())))

        scores_string = " ".join(final_dict.values())
        f.write("SCORE:     %s        %s\n"%(scores_string, tag))

def insert_Rosetta_chainbreaks( pose, binderlen ) -> core.pose.Pose:
    '''
    Given a pose and a list of indices to insert chainbreaks after,
    insert the chainbreaks into the pose.

    Args:
        pose (Pose)      : The pose to insert chainbreaks into.

        binderlen (list) : The length of the binder chain
    '''

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

def check_residue_distances(all_positions, all_positions_mask, max_amide_distance) -> list:
    '''
    Given a list of residue positions and a maximum amide distance, determine which residues
    are too far apart and should have a chainbreak inserted between them.

    This is mostly taken from the AF2 source code and modified for our purposes.
    '''

    breaks = []
    
    c_position = residue_constants.atom_order['C']
    n_position = residue_constants.atom_order['N']
    prev_is_unmasked = False
    this_c = None
    for i, (coords, mask) in enumerate(zip(all_positions, all_positions_mask)):

        # These coordinates only should be considered if both the C and N atoms are present.
        this_is_unmasked = bool(mask[c_position]) and bool(mask[n_position])
        if this_is_unmasked:
            this_n = coords[n_position]
            # Check whether the previous residue had both C and N atoms present.
            if prev_is_unmasked:

                distance = np.linalg.norm(this_n - prev_c)
                if distance > max_amide_distance:
                    # If the distance between the C and N atoms is too large, insert a chainbreak.
                    # This chainbreak is listed as being at residue i in zero-indexed numbering.
                    breaks.append(i)
                    print( f'The distance between residues {i} and {i+1} is {distance:.2f} A' +
                        f' > limit {max_amide_distance} A.' )
                    print( f"I'm going to insert a chainbreak after residue {i}" )

            prev_c = coords[c_position]

        prev_is_unmasked = this_is_unmasked

    return breaks

def subset_rmsd(
        xyz1: np.ndarray,
        align1: np.ndarray,
        calc1: np.ndarray,
        xyz2: np.ndarray,
        align2: np.ndarray,
        calc2: np.ndarray,
        eps=1e-6
    ) -> float:
    '''
        A general function to calculate the RMSD of a subset of atoms. This takes two sets of coordinates
        and aligns them on the subset of atoms defined by align1 and align2. It then calculates the RMSD
        of the subset of atoms defined by calc1 and calc2.

        Args:
            xyz1   : The first set of coordinates [L, 3]
            align1 : The indices of the atoms to align on in xyz1 [N]
            calc1  : The indices of the atoms to calculate the RMSD on in xyz1 [M]
            xyz2   : The second set of coordinates [L', 3]
            align2 : The indices of the atoms to align on in xyz2 [N]
            calc2  : The indices of the atoms to calculate the RMSD on in xyz2 [M]
            eps    : A small number to avoid dividing by zero

        Returns:
            rmsd   : The RMSD of the subset of atoms defined by calc1 and calc2

    '''

    assert(xyz1[align1].shape == xyz2[align2].shape), "The atoms to align on must be the same shape"
    assert(xyz1[calc1].shape == xyz2[calc2].shape), "The atoms to calculate the RMSD on must be the same shape"

    # center to CA centroid of the atoms to align on
    xyz1 = xyz1 - xyz1[align1].mean(0)
    xyz2 = xyz2 - xyz2[align2].mean(0)

    # Computation of the covariance matrix
    C = xyz2[align2].T @ xyz1[align1]

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate all of xyz2
    xyz2_ = xyz2 @ U

    assert(xyz2_[calc2].shape[1] == 3), "The last dimension of the prediction must be the 3 Cartesian coordinates"

    divL = xyz2_[calc2].shape[0]
    rmsd = np.sqrt(np.sum((xyz2_[calc2]-xyz1[calc1])*(xyz2_[calc2]-xyz1[calc1]), axis=(0,1)) / (divL + eps))

    return rmsd

def calculate_rmsds(
            init_crds : np.ndarray,
            pred_crds : np.ndarray,
            tmask     : np.ndarray
        ) -> dict:
    '''
        Given the initial coordinates and the predicted coordinates, calculate the Ca RMSD of the binders aligned
        on one another (binder_aligned_rmsd) and the Ca RMSD of the predicted binder aligned on the target (target_aligned_rmsd).

        Args:

            init_crds : The initial coordinates of the complex [L, 27, 3]

            pred_crds : The predicted coordinates of the complex [L, 27, 3]

            tmask     : A mask indicating which residues are part of the target chain [L]

        Returns:

            rmsds     : A dictionary containing the RMSDs of the binder aligned on the binder and the binder aligned on the target
    
    '''

    rmsds = {}

    init_ca = init_crds[:, 1, :]
    pred_ca = pred_crds[:, 1, :]

    rmsds['binder_aligned_rmsd'] = subset_rmsd(
        xyz1   = init_ca,
        align1 = ~tmask,
        calc1  = ~tmask,
        xyz2   = pred_ca,
        align2 = ~tmask,
        calc2  = ~tmask
    )

    rmsds['target_aligned_rmsd'] = subset_rmsd(
        xyz1   = init_ca,
        align1 = tmask,
        calc1  = ~tmask,
        xyz2   = pred_ca,
        align2 = tmask,
        calc2  = ~tmask
    )

    return rmsds

