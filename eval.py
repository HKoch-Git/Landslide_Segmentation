
############################################
## Files were changed from original!     ###
############################################


# Entire file/many imports originally taken directly and adapted/restructured from (including the structure of load_eval_model_data function)
# 6/8: https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_image_classifier.py

# Changes made to the code: Most of the logic inside cli_main, including function arguments, etc. The code taken was mostly the function structure and some template-like logic like seeding things, while arguments were changed.
# Some of the imports were likely added/removed.

import argparse
from argparse import ArgumentParser

from parsers import eval_parser
from classifier import SinkholeSegmenter

import pytorch_lightning as pl

import json
import os

from eval_utils import load_eval_modules, eval_funcs
import torch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
def cli_main():
    
    """
    Note that 'model' is really referring to a Classifier, not the raw model (i.e. U-Net) itself
    This was changed to strictly run on one data_mode per inference script.
        So will have to re-run everything, once on args: {val, test}
            This is to avoid breaking up the code (which runs faster, but can be brittle in refactor)
    """

    # ------------
    # args
    # ------------

    # 6/9: debugging help https://github.com/tensorflow/tensorflow/issues/8389
    # Not a specific code citation.

    # 6/8: Guidance on how to manage options, as well as organization structure for parsers:
# https://github.com/NVlabs/SPADE/blob/master/options/base_options.py
    parser_obj = eval_parser.EvalParser()
    parser = parser_obj.initialize(ArgumentParser())
    #end SPADE code
    
    this_args = vars(parser_obj.parse())
    
    # Below: 
    
    print('NOTE: Remove check for seed in args in eval.py once everything from the first run is trained.')
    
    pl.seed_everything(this_args['seed'] if 'seed' in this_args else 1234, workers = True)

    # Need to read in the information from the config file, and add them to args
    model, datasets, dataloaders = load_eval_modules.load_eval_model_data(this_args)
    
    print('May require re-factoring slightly for inference script -- see the comments at around this print statement.')

    data_loader = dataloaders[this_args['phase']]
    
    assert data_loader.dataset.eval_pad, "Code was designed to run eval code with is_eval_mode due to padding."
        
    # Note to self unless future is adjusted?
    #assert this_args['model_type'] == 'unet', "Have not yet updated code to accept alternate models, will do this once new architectures are proposed."
    
    metrics_results = eval_funcs.evaluate_metrics(this_args, datasets,dataloaders, model)
    
if __name__ == '__main__':
    cli_main()
    
    
# end structure/code of entire file taken from the template code