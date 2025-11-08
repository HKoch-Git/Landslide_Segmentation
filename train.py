
############################################
## Files were changed from original!     ###
############################################


# Entire file/pl-related imports originally taken directly and adapted from:
# 6/8: https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_image_classifier.py

# Changes made to the code: Most of the logic inside cli_main, including function arguments, etc. The code taken was mostly the function structure and some template-like logic like seeding things, while arguments were changed.
# Some of the imports were likely added/removed.


from parsers import train_parser
from classifier import SinkholeSegmenter

import pytorch_lightning as pl

# 6/8: https://github.com/NVlabs/SPADE/blob/master/options/base_options.py
# taken imports
import argparse
from argparse import ArgumentParser
# end taken imports

import data_factory_new

from utils import exec_file
import json

import os
import torch

#handle densenet encoders certificate error
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def cli_main():
    
    # 6/9 (not a code cite, just a reference): debugging help https://github.com/tensorflow/tensorflow/issues/8389

    # 6/8: Guidance on how to manage options, as well as organization structure for parsers:
# https://github.com/NVlabs/SPADE/blob/master/options/base_options.py
    this_parser_obj = train_parser.TrainParser()
    parser = this_parser_obj.initialize(ArgumentParser())
    
    raw_args = this_parser_obj.parse()
    # end SPADE guidance/code
    
    args = vars(raw_args)


    
    pl.seed_everything(args['seed'], workers = True)
    
    # Tuple is expected throughout the rest of the original code.
    
    datasets, dataloaders = data_factory_new.get_all_data(args, is_eval_mode = False)
    print('Number of traning patchs is: ',len(datasets['train']))
    
    kwargs = exec_file.get_model_kwargs(args)
    
    this_trainer, true_save_path, this_callback = exec_file.configure_trainer(raw_args)

    model = SinkholeSegmenter(**kwargs)
    
    this_trainer.fit(model, dataloaders['train'], dataloaders['val'])
    
    # Dump the config file into the right folder, so it can be loaded in the eval mode.
    # 6/10: https://www.kite.com/python/answers/how-to-dump-a-dictionary-to-a-json-file-in-python
    
    # Note best checkpoint path for use in evaluation
    
    args['checkpoint_path'] = this_callback.best_model_path
    with open(os.path.join(true_save_path, 'config.json'), 'w') as f: 
        json.dump(args, f)
        
    # end kite code
    print('Saved configuration files to: ', os.path.join(true_save_path, 'config.json'))

if __name__ == '__main__':
    cli_main() 
    
    
# end entire file taken from the template code.