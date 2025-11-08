
import os
import glob

from pprint import pprint


def get_eval_out_dir(args):
    return os.path.dirname(args['checkpoint_path'])

def get_experiment_folder(checkpoint_path):
    """
    This is because the experiment is specified via the checkpoint file itself.
    Want the parent folder of the parent folder of the file for the experiment folder.
    """
    return os.path.dirname(os.path.dirname(checkpoint_path))


def get_experiment_matching_string(input_data_type, norm_type):
    """
    Returns the section of an experiment path that would correspond to a given config
        of input type and norm type
    """
    return '/'.join([input_data_type] + [norm_type for _ in range(5)])

def retrieve_results(root_path):
    
    # Code to get the paths, 6/16:
    # https://stackoverflow.com/questions/18394147/recursive-sub-folder-search-and-return-files-in-a-list-python
    
    all_paths = [y for x in os.walk(root_path) for y in glob.glob(os.path.join(x[0], '*.ckpt'))]
    
    # end paths code
    
    return sorted(all_paths)
     
def find_matching_checkpoint_path(experiment_path_partial, all_experiment_paths):
    
    """
    Find the appropriate checkpoints based on its path components.
    Note that this assumes (and checks) that there is one checkpoint
        per experiment normalization/data type 'configuration',
        which is what would result from running the replication code.
    
    all_experiment_paths holds checkpoints, NOT experiment folders
        because it's generally calculated via retrieve_results.
    """
    
    has_vars = lambda path : experiment_path_partial in path
    all_matches = list(filter(has_vars, all_experiment_paths))
    
    assert len(all_matches) == 1, 'Either zero or multiple checkpoints per experimental folder (defined as config. of data type and norm)'
    
    # Get the checkpoint, then revert up to the experiment path itself
    exp_checkpoint_path = all_matches[0]
    
    return exp_checkpoint_path



def find_matching_experiment_path(experiment_path_partial, all_experiment_paths):
    
    checkpoint = find_matching_checkpoint_path(experiment_path_partial, all_experiment_paths)
    return get_experiment_folder(checkpoint)
    
    
    
def find_matching_checkpoint_path_by_attr(data_type, norm_type, root_path):
    
    all_paths = retrieve_results(root_path)
    experiment_name_find = get_experiment_matching_string(data_type, norm_type)
    
    return find_matching_checkpoint_path(experiment_name_find, all_paths)
    
    
    
    
def find_matching_experiment_path_by_attr(data_type, norm_type, root_path):
    
    """
    Finds based on the data type and the norm type, from a root folder.
    """
    checkpoint = find_matching_checkpoint_path_by_attr(data_type, norm_type, root_path)
    return get_experiment_folder(checkpoint)
    
    
    
    