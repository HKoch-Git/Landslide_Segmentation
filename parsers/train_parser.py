# 6/8: Organization, imports, and all initialization code/class structure taken from:
# https://github.com/NVlabs/SPADE/blob/master/options/base_options.py (for the code/class definition)
# https://github.com/NVlabs/SPADE/blob/master/train.py (for the hierarchical structure)

# Modifications made to code: Almost all logic in the file, except for the logic/code described above. Lines/function organization was taken in pieces from the original files.

import argparse

from tensorboard.compat.tensorflow_stub.dtypes import double

from parsers import gen_parser

import os

class TrainParser(gen_parser.GenParser):
    
    # Training data-loading and computing arguments.
    
    def initialize(self, parser):
        
        parser.add_argument('--dataset_name', type = str, default = 'ky', help = 'Name of dataset for training and evaluation. Base directory and contents specified in filenames_config.')
        
        parser.add_argument('--seed', type = int, default = 1234, help = 'Random seed to use.')
        
        parser.add_argument('--model_type', type = str, default = 'unet', help = "Model architecture options: unet, unetpp,FPN,dl3p,manet")
        
        parser.add_argument('--norm_type', type = str, default = 'batch', help = 'Type of normalization layer: batch, instance')
        
        parser.add_argument('--input_type', type = str, default = 'dem_ddxy', help = 'Type of model inputs: dem, shaded_relief, naip, dem_ddxy, dem_dxy_pre, spp')
        parser.add_argument('--normalize_dem', type = str, default = 'none', help = 'What type of normalization for DEM input: 0_to_1, unit_gaussian, instance, none ')
        parser.add_argument('--normalize_shaded', type = str, default = '0_to_1', help = 'What type of normalization for shaded input: 0_to_1, unit_gaussian, instance')
        parser.add_argument('--normalize_naip', type = str, default = 'none', help = 'What type of normalization for NAIP input: 0_to_1, unit_gaussian, instance')
        parser.add_argument('--normalize_dem_ddxy', type = str, default = 'unit_gaussian', help = 'What type of normalization for dem gradients input: instance, none, 0_to_1, unit_gaussian')
        parser.add_argument('--normalize_dem_dxy_pre', type = str, default = 'none', help = 'What type of normalization for dem slope input: instance, none, 0_to_1, unit_gaussian')
        parser.add_argument('--normalize_spp', type = str, default = 'unit_gaussian', help = 'What type of normalization for slope-plan-prof-curvs input: instance, none, 0_to_1, unit_gaussian')
        
        parser.add_argument('--feature_weight', type = float, default = 20.0, help = 'What factor to weight the feature class by')
        parser.add_argument('--encoder', type = str, default = 'resnet34', help = 'specify encoder')

        ###### parser.add_argument('--train_width_percent', type = float, default = 1.0, help = 'What percentage of each input width is partitioned as train.')
        
        # Training arguments
        
        # 6/9: From the warning in the Terminal output from PytorchLightning
        parser.add_argument('--gpus', type = int, default = 1, help = 'Number of GPUs to train on')
        # end recommendation of warning
        
        parser.add_argument('--train_split_orientation', choices = ['left', 'right', 'top', 'bottom'], default = 'top', help='Which side of the data image does the training section lie along')

        # This train height percentage may be integrated later. To be safe I'm staying with the old trainval_split for now.
        #bring back train height percentage option 11/8/2022 jZ
        #parser.add_argument('--train_height_percent', type = bool, default = 0.5, help = 'What percentage of each input height is partitioned as train. The remaining portions correspond to val and test -- see diagram in paper.')
        ###### parser.add_argument('--train_height_percent', type = float, default = 0.75, help = 'What percentage of each input height is partitioned as train. The remaining portions correspond to val and test -- see diagram in paper.')
        
        # https://stackoverflow.com/a/12117065 
        def float_in_range(lo, hi):
            def f(s):
                try:
                    x = float(s)
                except ValueError:
                    raise argparse.ArgumentTypeError(f'{x} is not a float literal')
                
                if x < lo or x > hi:
                    raise argparse.ArgumentTypeError(f'{x} is not in the range [{lo}, {hi}]')
                
                return x
            return f

        parser.add_argument('--train_split', type=float_in_range(0.5,0.95), default=0.75, help = 'Split of whole image between train and non-train')

        #add an option for val/test split
        parser.add_argument('--val_test_split', type = float, default = 0.4, help="Split of non-train image between validation and test")
        
        parser.add_argument('--cutout_size', type = int, default = 400, help = 'Final image size. Input dim, such that resultant dimensions are (dim, dim).')
        # Note: Only tested for 400 so far
        parser.add_argument('--batch_size', type = int, default = 14)
        parser.add_argument('--learning_rate', type = float, default = 5e-4, help = 'Initial learning rate')
        parser.add_argument('--weight_decay', type = float, default = 1e-6)
        parser.add_argument('--lr_decay', type = float, default = 0.9)
        parser.add_argument('--lr_decay_every', type = float, default = 3)
        parser.add_argument('--shuffle', type = bool, default = True)
        parser.add_argument('--max_epochs', type = int, default = 100) 
        #parser.add_argument('--max_epochs', type = int, default = 2) # While still developing the code.
        
        #parser.add_argument('--num_workers', type = int, default = 8, help = 'Workers for dataloading')
        parser.add_argument('--num_workers', type = int, default = 2, help = 'Workers for dataloading')
        parser.add_argument('--default_root_dir', type = str, default = './records', help = 'Directory to save the logs and checkpoints.')
        parser.add_argument("--accelerator", type=str, default="gpu")
        parser.add_argument('--deterministic', type=bool, default=False)
        return parser

