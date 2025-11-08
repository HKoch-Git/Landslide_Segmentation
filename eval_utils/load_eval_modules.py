


############################################
## Files were changed from original!     ###
############################################


# This file was originally taken and adapted from
# 6/8: https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_image_classifier.py

# In particular structure of loading dataloaders, models, etc.

# In terms of revisiting the source, however, this has diverged from the original code significantly (nearly all lines are different)
# to where there are not obvious places to isolate for a cite,
# but I maintain the cite to note the very first structure that I used.


import os
from os.path import join, exists

import json
from classifier import SinkholeSegmenter
from utils import path_utils

import data_factory_new
from data_utils import get_raw_stats_data

import filenames_config

from eval_utils import eval_funcs


def load_best_threshold(args, dataloaders, model):
    
    out_dir = path_utils.get_eval_out_dir(args)
    
    threshold_path = join(out_dir, 'best_threshold.json')
    
    # Always use threshold based on validation data. 
    
    if args['phase'] in {'val', 'test'}:
        
        best_threshold = eval_funcs.find_best_threshold(args, 'val', dataloaders, model) 
        with open(threshold_path, 'w') as f:
            # 6/21/21 https://www.geeksforgeeks.org/json-dump-in-python/
            json.dump({'best_threshold' : best_threshold}, f)
            # end cite
        
    else: # Inference mode
        
        with open(threshold_path, 'r') as f:
            best_threshold = json.load(f)['best_threshold']
            
            print('Loading threshold from json file')
            
    return best_threshold
    
    
def get_eval_modules(args):
    
    old_checkpoint_path = args['checkpoint_path']
    
    exp_path = path_utils.get_experiment_folder(old_checkpoint_path)
    print(f'\nLoading evaluation modules for the following experiment: {exp_path}\n')

    with open(os.path.join(exp_path, 'config.json'), 'r') as f: 
        
        new_args = json.load(f)
        
        if 'input_type' in args:
            if args['input_type'] != new_args['input_type']:
                assert False, "Trying to either run eval.py or more likely inference_main.py with an inference input type that is not the same as the checkpoint's input_type."
            
        args.update(new_args) # All of the config from the train save.
        
        # IMPORTANT: Overriding certain arguments for evaluation mode. Matches original repository.
        args['batch_size'] = 1
        args['shuffle'] = False
        args['num_workers'] = 0 
        
    args['train_dataset_name'] = args['dataset_name'] # Change these
    del(args['dataset_name'])
    
    args['checkpoint_path'] = old_checkpoint_path
    
    
    # Above: Overwrite, don't use the checkpoint path in the .json due to relative path issues, see Issue 20 for more details.
    
    
    # 6/10: https://pytorch-lightning.readthedocs.io/en/stable/common/weights_loading.html
    # Loads parameters and hyperparams
    model = SinkholeSegmenter.load_from_checkpoint(args['checkpoint_path']) 
    # end this snippet
    model = model.eval() # Needed because you're not going to use the traditional validation loop in evaluation phase.
    
    return args, model
    
    
def load_inference_model_data(args):
    
    args['phase'] = 'inference'
    
    args, model = get_eval_modules(args)

    datasets, dataloaders = data_factory_new.get_inference_data(args)
    
    return model, datasets, dataloaders
    
    

def load_eval_model_data(partial_args):
    """
    Function for loading model, datasets, and dataloaders as specified by an experiment/checkpoint path.
    It's called "checkpoint_path" in the args but it's really a checkpoint folder.
    """
    
    args, model = get_eval_modules(partial_args)
    datasets, dataloaders = data_factory_new.get_all_data(partial_args, is_eval_mode = True)
    
    return model, datasets, dataloaders

# end adapted code for both sources
