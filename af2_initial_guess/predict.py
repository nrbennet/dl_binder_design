#!/usr/bin/env python

import os
import numpy as np
import sys

from timeit import default_timer as timer
import argparse
import glob
import uuid

import jax
import jax.numpy as jnp

from jax.lib import xla_bridge

from alphafold.common import residue_constants
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.data import pipeline
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model

import af2_util

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parent, 'include'))
from silent_tools import silent_tools

from pyrosetta import *
from rosetta import *
init( '-in:file:silent_struct_type binary -mute all' )

def range1(size): return range(1, size+1)

#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()

# I/O Arguments
parser.add_argument( "-pdbdir", type=str, default="", help='The name of a directory of pdbs to run through the model' )
parser.add_argument( "-silent", type=str, default="", help='The name of a silent file to run through the model' )
parser.add_argument( "-outpdbdir", type=str, default="outputs", help='The directory to which the output PDB files will be written. Only used when -pdbdir is active' )
parser.add_argument( "-outsilent", type=str, default="out.silent", help='The name of the silent file to which output structs will be written. Only used when -silent is active' )
parser.add_argument( "-runlist", type=str, default='', help="The path of a list of pdb tags to run. Only used when -pdbdir is active (default: ''; Run all PDBs)" )
parser.add_argument( "-checkpoint_name", type=str, default='check.point', help="The name of a file where tags which have finished will be written (default: check.point)" )
parser.add_argument( "-scorefilename", type=str, default='out.sc', help="The name of a file where scores will be written (default: out.sc)" )
parser.add_argument( "-maintain_res_numbering", action="store_true", default=False, help='When active, the model will not renumber the residues when bad inputs are encountered (default: False)' )

parser.add_argument( "-debug", action="store_true", default=False, help='When active, errors will cause the script to crash and the error message to be printed out (default: False)')

# AF2-Specific Arguments
parser.add_argument( "-max_amide_dist", type=float, default=3.0, help='The maximum distance between an amide bond\'s carbon and nitrogen (default: 3.0)' )
parser.add_argument( "-recycle", type=int, default=3, help='The number of AF2 recycles to perform (default: 3)' )
parser.add_argument( "-no_initial_guess", action="store_true", default=False, help='When active, the model will not use an initial guess (default: False)' )
parser.add_argument( "-force_monomer", action="store_true", default=False, help='When active, the model will predict the structure of a monomer (default: False)' )

args = parser.parse_args()

class FeatureHolder():
    '''
    This is a struct which holds the features for a single structure being run through the model
    '''

    def __init__(self, pose, monomer, binderlen, tag):
        self.pose   = pose
        self.tag    = tag
        self.outtag = self.tag + '_af2pred'
        
        self.seq       = pose.sequence()
        self.binderlen = binderlen
        self.monomer   = monomer
        
        # Pre model features
        self.initial_all_atom_positions = None
        self.initial_all_atom_masks = None

        # Post model features
        self.outpose     = None
        self.plddt_array = None
        self.score_dict  = None

class AF2_runner():
    '''
    This class handles generating features, running the model, and parsing outputs
    '''

    def __init__(self, args, struct_manager):

        self.max_amide_dist = args.max_amide_dist

        # For timing
        self.t0 = None

        self.struct_manager = struct_manager

        # Other models may be run but their weights will also need to be downloaded
        self.model_name = "model_1_ptm"

        model_config = config.model_config(self.model_name)
        model_config.data.eval.num_ensemble = 1

        model_config.data.common.num_recycle = args.recycle
        model_config.model.num_recycle = args.recycle

        model_config.model.embeddings_and_evoformer.initial_guess = False if args.no_initial_guess else True

        model_config.data.common.max_extra_msa = 5
        model_config.data.eval.max_msa_clusters = 5

        params_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'model_weights')

        model_params = data.get_model_haiku_params(model_name=self.model_name, data_dir=params_dir)

        self.model_runner = model.RunModel(model_config, model_params)

    def featurize(self, feat_holder) -> None:

        all_atom_positions, all_atom_masks = af2_util.af2_get_atom_positions(feat_holder.pose, self.struct_manager.tmp_fn)

        feat_holder.initial_all_atom_positions = all_atom_positions
        feat_holder.initial_all_atom_masks     = all_atom_masks

        initial_guess = af2_util.parse_initial_guess(feat_holder.initial_all_atom_positions)

        # Determine which residues to template
        if feat_holder.monomer:
            # For monomers predict all residues
            feat_holder.residue_mask = [False for i in range(len(feat_holder.seq))]
        else:
            # For interfaces fix the target and predict the binder
            feat_holder.residue_mask = [int(i) > feat_holder.binderlen for i in range(len(feat_holder.seq))]

        template_dict = af2_util.generate_template_features(
                                                            feat_holder.seq,
                                                            feat_holder.initial_all_atom_positions,
                                                            feat_holder.initial_all_atom_masks,
                                                            feat_holder.residue_mask
                                                           )
        # Gather features
        feature_dict = {
            **pipeline.make_sequence_features(sequence=feat_holder.seq,
                                            description="none",
                                            num_res=len(feat_holder.seq)),
            **pipeline.make_msa_features(msas=[[feat_holder.seq]],
                                        deletion_matrices=[[[0]*len(feat_holder.seq)]]),
            **template_dict
        }

        if feat_holder.monomer:
            breaks = []
        else:
            breaks = af2_util.check_residue_distances(
                            feat_holder.initial_all_atom_positions,
                            feat_holder.initial_all_atom_masks,
                            self.max_amide_dist
                        )

        feature_dict['residue_index'] = af2_util.insert_truncations(feature_dict['residue_index'], breaks)

        feature_dict = self.model_runner.process_features(feature_dict, random_seed=0)

        return feature_dict, initial_guess 

    def generate_scoredict(self, feat_holder, confidences, rmsds) -> None:
        '''
        Collect the confidence values, slicing them to the binder and target regions
        then add the parsed scores to the score_dict
        '''

        binderlen = feat_holder.binderlen

        plddt_array = confidences['plddt']
        plddt = np.mean( plddt_array )

        if feat_holder.monomer:
            plddt_binder = np.mean( plddt_array )
            plddt_target = float('nan')
        else:
            plddt_binder = np.mean( plddt_array[:binderlen] )
            plddt_target = np.mean( plddt_array[binderlen:] )

        pae = confidences['predicted_aligned_error']

        if feat_holder.monomer:
            pae_binder = np.mean( pae )
            pae_target = float('nan')
            pae_interaction_total = float('nan')
        else:
            pae_interaction1 = np.mean( pae[:binderlen,binderlen:] )
            pae_interaction2 = np.mean( pae[binderlen:,:binderlen] )
            pae_binder = np.mean( pae[:binderlen,:binderlen] )
            pae_target = np.mean( pae[binderlen:,binderlen:] )
            pae_interaction_total = ( pae_interaction1 + pae_interaction2 ) / 2

        time = timer() - self.t0

        score_dict = {
                "plddt_total" : plddt,
                "plddt_binder" : plddt_binder,
                "plddt_target" : plddt_target,
                "pae_binder" : pae_binder,
                "pae_target" : pae_target,
                "pae_interaction" : pae_interaction_total,
                "binder_aligned_rmsd": rmsds['binder_aligned_rmsd'],
                "target_aligned_rmsd": rmsds['target_aligned_rmsd'],
                "time" : time
        }

        # Store this in the feature holder for later use
        feat_holder.score_dict = score_dict

        # If we ever want to write strings to the score file we can do it here
        string_dict = None

        self.struct_manager.record_scores(feat_holder.outtag, score_dict, string_dict)

        print(score_dict)
        print(f"Tag: {feat_holder.outtag} reported success in {time} seconds")

    def process_output(self, feat_holder, feature_dict, prediction_result) -> None:
        '''
        Take the AF2 output, parse the confidence scores from this and register the scores in the score file
        Also write out the structure
        '''

        # First extract the structure and confidence scores from the prediction result
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
        
        feat_holder.plddt_array = confidences['plddt']

        # Calculate the RMSDs
        target_mask = np.zeros(len(feat_holder.seq), dtype=bool)
        target_mask[feat_holder.binderlen:] = True

        rmsds = af2_util.calculate_rmsds(
            feat_holder.initial_all_atom_positions,
            this_protein.atom_positions,
            target_mask
        )

        # Write the structure as a pdb file so Rosetta can read it
        unrelaxed_pdb_lines = protein.to_pdb(this_protein)
        
        with open(self.struct_manager.tmp_fn, 'w') as f: f.write(unrelaxed_pdb_lines)

        # Now read the structure to a Rosetta pose
        feat_holder.outpose = pyrosetta.pose_from_file(self.struct_manager.tmp_fn)

        os.remove(self.struct_manager.tmp_fn)
        
        # Now we can finally write the scores and the predicted structure to disk
        self.generate_scoredict(feat_holder, confidences, rmsds)
        self.struct_manager.dump_pose(feat_holder)
    
    def process_struct(self, tag) -> None:

        # Start the timer
        self.t0 = timer()

        # Load the structure
        pose, monomer, binderlen, usetag = self.struct_manager.load_pose(tag)

        # Store the pose in the feature holder
        feat_holder = FeatureHolder(pose, monomer, binderlen, usetag)

        print(f'Processing struct with tag: {feat_holder.tag}')

        # Generate features
        feature_dict, initial_guess = self.featurize(feat_holder)

        # Run model
        start = timer()
        print(f'Running {self.model_name}')

        prediction_result = self.model_runner.apply( self.model_runner.params,
                                                     jax.random.PRNGKey(0),
                                                     feature_dict,
                                                     initial_guess )

        print(f'Tag: {feat_holder.tag} finished AF2 prediction in {timer() - start} seconds')

        # Process outputs
        self.process_output(feat_holder, feature_dict, prediction_result)

class StructManager():
    '''
    This class handles all of the input and output for the AF2 model. It deals with silent files vs. pdbs,
    checkpointing, and writing of outputs

    Note: This class could be moved to a separate file and shared with ProteinMPNN
    '''

    def __init__(self, args):
        self.args = args

        self.maintain_res_numbering = args.maintain_res_numbering

        self.score_fn = args.scorefilename

        # Generate a random unique temporary filename
        self.tmp_fn = f'tmp_{uuid.uuid4()}.pdb'

        self.force_monomer = args.force_monomer

        self.silent = False
        if not args.silent == '':
            self.silent = True

            self.struct_iterator = silent_tools.get_silent_index(args.silent)['tags']

            self.sfd_in = rosetta.core.io.silent.SilentFileData(rosetta.core.io.silent.SilentFileOptions())
            self.sfd_in.read_file(args.silent)

            self.sfd_out = core.io.silent.SilentFileData(args.outsilent, False, False, "binary", core.io.silent.SilentFileOptions())

            self.outsilent = args.outsilent

        self.pdb = False
        if not args.pdbdir == '':
            self.pdb = True

            self.pdbdir    = args.pdbdir
            self.outpdbdir = args.outpdbdir

            self.struct_iterator = glob.glob(os.path.join(args.pdbdir, '*.pdb'))

            # Parse the runlist and determine which structures to process
            if args.runlist != '':
                with open(args.runlist, 'r') as f:
                    self.runlist = set([line.strip() for line in f])

                    # Filter the struct iterator to only include those in the runlist
                    self.struct_iterator = [struct for struct in self.struct_iterator if '.'.join(os.path.basename(struct).split('.')[:-1]) in self.runlist]

                    print(f'After filtering by runlist, {len(self.struct_iterator)} structures remain')

        # Assert that either silent or pdb is true, but not both
        assert(self.silent ^ self.pdb), f'Both silent and pdb are set to {args.silent} and {args.pdb} respectively. Only one of these may be active at a time'

        # Setup checkpointing
        self.chkfn = args.checkpoint_name
        self.finished_structs = set()

        if os.path.isfile(self.chkfn):
            with open(self.chkfn, 'r') as f:
                for line in f:
                    self.finished_structs.add(line.strip())

    def input_check(self, pose, tag) -> bool:
        '''
        This function checks that the given pose is valid for AF2. It specifically
        checks that all residue indices are unique
        '''
        
        seen_indices = set()
        
        # Loop through the PDBinfo of the pose and check that all residue indices are unique
        pdbinfo = pose.pdb_info()
        for resi in range1(pose.size()):
            residx = pdbinfo.number(resi)
            if residx in seen_indices:
                print( f"\nNon-unique residue indices detected for tag: {tag}. " +
                        "This will cause AF2 to yield garbage outputs." )
                return False

            seen_indices.add(residx)
        
        return True

    def record_checkpoint(self, tag):
        '''
        Record the fact that this tag has been processed.
        Write this tag to the list of finished structs
        '''
        with open(self.chkfn, 'a') as f:
            f.write(f'{tag}\n')

    def iterate(self):
        '''
        Iterate over the silent file or pdb directory and run the model on each structure
        '''

        # Iterate over the structs and for each, check that the struct has not already been processed
        for struct in self.struct_iterator:
            if self.pdb:
                tag = '.'.join(os.path.basename(struct).split('.')[:-1])
            else:
                tag = struct    

            if tag in self.finished_structs:
                print(f'{tag} has already been processed. Skipping')
                continue

            yield struct

    def record_scores(self, tag, score_dict, string_dict):
        '''
        Record the scores for this structure to the score file.

        Args:
            tag (str): The tag for this structure
            score_dict (dict): A dictionary of numerical scores to record
            string_dict (dict): A dictionary of string scores to record
        '''

        # Check whether the score file exists
        write_header = False
        if not os.path.isfile(self.score_fn):
            write_header = True

        af2_util.add2scorefile(tag, self.score_fn, write_header, score_dict, string_dict) 

    def dump_pose(self, feat_holder):
        '''
        Dump this pose to either a silent file or a pdb file depending on the input arguments
        '''
        
        if feat_holder.monomer:
            pose = feat_holder.outpose
        else:
            # Insert chainbreaks into the pose
            pose = af2_util.insert_Rosetta_chainbreaks(feat_holder.outpose, feat_holder.binderlen)

        # Add the plddt scores as b-factors to the pose
        info = pose.pdb_info()
        for resi in range1(pose.size()):
            info.add_reslabel(resi, f'{feat_holder.plddt_array[resi-1]}')
            for atom_i in range1(pose.residue(resi).natoms()):
                info.bfactor(resi, atom_i, feat_holder.plddt_array[resi-1])
        
        # Assign the pose the updated pdb_info
        pose.pdb_info(info)

        if self.pdb:
            # If the outpdbdir does not exist, create it
            # If there are parents in the path that do not exist, create them as well
            if not os.path.exists(self.outpdbdir):
                os.makedirs(self.outpdbdir)

            pdbfile = os.path.join(self.outpdbdir, feat_holder.outtag + '.pdb')
            pose.dump_pdb(pdbfile)
        
        if self.silent:
            struct = self.sfd_out.create_SilentStructOP()
            struct.fill_struct(pose, feat_holder.outtag)

            # Write the scores to the silent file
            for scorename, value in feat_holder.score_dict.items():
                struct.add_energy(scorename, value, 1)

            self.sfd_out.add_structure(struct)
            self.sfd_out.write_silent_struct(struct, self.outsilent)

    def load_pose(self, tag):
        '''
        Load a pose from either a silent file or a pdb file depending on the input arguments.

        Also run input checking on the pose to ensure that it is valid for AF2. The pose
        will be automatically renumbered if -maintain_res_numbering is not False
        '''

        if not self.pdb and not self.silent:
            raise Exception('Neither pdb nor silent is set to True. Cannot load pose')

        if self.pdb:
            pose = pose_from_pdb(tag)
            usetag = '.'.join(os.path.basename(tag).split('.')[:-1])
        
        if self.silent:
            pose = Pose()
            usetag = tag
            self.sfd_in.get_structure(tag).fill_pose(pose)
        
        # Run input checking on the pose
        passes_check = self.input_check(pose, tag)

        if not passes_check:
            if not self.maintain_res_numbering:
                print( f"Renumbering {tag}" )
                
                info = core.pose.PDBInfo(pose)
                pose.pdb_info(info)
            else:
                raise Exception( f"Pose {tag} failed input checking.")

        # Determine whether we are working with a monomer or a complex
        splits = pose.split_by_chain()
        if len(splits) > 2:
            raise Exception( f"Pose {tag} has more than two chains. This is not supported by this script." )

        elif len(splits) < 1:
            raise Exception( f"Pose {tag} is empty. This is not supported by this script." )

        elif len(splits) == 1:
            monomer   = True
            binderlen = -1

        elif len(splits) == 2:
            if self.force_monomer:
                print( "/" * 60 ) 
                print( f"Pose {tag} has two chains. But force_monomer is set to True. Treating as monomer.")
                print( f"I am going to assume that the first chain is the binder and that is the chain I will predict")
                print( "/" * 60 ) 

                monomer   = True
                pose      = splits[1]
                binderlen = -1
            else: 
                monomer   = False
                binderlen = splits[1].size()

        return pose, monomer, binderlen, usetag

####################
####### Main #######
####################

device = xla_bridge.get_backend().platform
if device == 'gpu':
    print('/' * 60)
    print('/' * 60)
    print('Found GPU and will use it to run AF2')
    print('/' * 60)
    print('/' * 60)
    print('\n')
else:
    print('/' * 60)
    print('/' * 60)
    print('WARNING! No GPU detected running AF2 on CPU')
    print('/' * 60)
    print('/' * 60)
    print('\n')

struct_manager = StructManager(args)
af2_runner     = AF2_runner(args, struct_manager)

for pdb in struct_manager.iterate():

    if args.debug: af2_runner.process_struct(pdb)

    else: # When not in debug mode the script will continue to run even when some poses fail
        t0 = timer()

        try: af2_runner.process_struct(pdb)

        except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )

        except:
            seconds = int(timer() - t0)
            print( "Struct with tag %s failed in %i seconds with error: %s"%( pdb, seconds, sys.exc_info()[0] ) )

    # We are done with one pdb, record that we finished
    struct_manager.record_checkpoint(pdb)
