# Useful functions for running train and eval.py

# For import in logging
# 6/10: https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
# end imports

import os
from datetime import datetime

# 6/15: for below import
# https://pytorch-lightning.readthedocs.io/en/stable/common/weights_loading.html

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# end import

def get_model_kwargs(args_dict):
    
    """
    For initializing the SinkholeSegmenter
    """
    
    segmenter_args = ['input_type', 'model_type', 'norm_type', 'gpus', 'learning_rate', 'weight_decay', 'lr_decay_every', 'lr_decay', 'feature_weight','encoder']
    kwargs = {key : args_dict[key] for key in segmenter_args}
    return kwargs

def configure_trainer(raw_all_args):
    """
    Makes the right trainer and also returns the save path of the trainer.
    raw_all_args = the Namespace form of the args
    
    Use this for train/val phase, not val/test (evaluation). 
    """

    save_folder, save_time = get_reproducibility_string(raw_all_args)
    
    default_root_dir = raw_all_args.default_root_dir
    
    # 6/10: Logger code https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
    tb_logger = pl_loggers.TensorBoardLogger(save_dir = default_root_dir, name = save_folder, version = save_time)
    # end logger code

    # Information on default behavior: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.model_checkpoint.html
    # 6/15: Code to configure Trainer for "best"-checkpointing in the Trainer declaration below, also to make the checkpoint callback:
    # https://pytorch-lightning.readthedocs.io/en/stable/common/weights_loading.html

    
    #argument "every_n_val_epochs" is removed from modelcheckpoint, replace it with "every_n_epochs" by JZ 08/22/2022
    #checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode = 'min', every_n_val_epochs  = 1)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode = 'min', every_n_epochs  = 1)
    # end callback code
    
    # 7/20/21: Advice to use the LR tracking (not for code)
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/3795
    lr_callback = LearningRateMonitor() # Changed this after running the initial two trains, because the intervals were still irregular.
    # end advice
    
    # ===========
        
    # 6/8: https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_image_classifier.py
    # changed the arguments
    
    ############################################
    ## Files were changed from original!     ###
    ############################################
    
    # turn off deterministic JZ 08/22/2022
    #trainer = pl.Trainer.from_argparse_args(raw_all_args, logger=tb_logger, deterministic = True, callbacks=[lr_callback, checkpoint_callback])
    #trainer = pl.Trainer.from_argparse_args(raw_all_args, logger=tb_logger, deterministic = False, callbacks=[lr_callback, checkpoint_callback])
    trainer = pl.Trainer(accelerator=raw_all_args.accelerator, max_epochs=raw_all_args.max_epochs,
                         default_root_dir=raw_all_args.default_root_dir, logger=tb_logger, deterministic=raw_all_args.deterministic,
                         callbacks=[lr_callback, checkpoint_callback])
    # end taken from template code
    
    true_path = os.path.join(default_root_dir, os.path.join(save_folder, save_time))
    
    return trainer, true_path, checkpoint_callback

    
def get_reproducibility_string(raw_args):
        
        which_args = ['model_type', 'norm_type', 'input_type', 'normalize_dem', 'normalize_shaded', 'normalize_naip', 'normalize_dem_ddxy', 'normalize_dem_dxy_pre', 'train_split_orientation', 'train_split', 'dataset_name']
        
        # 6/9 : https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-argparse-namespace-as-a-dictionary
            
        these_args = vars(raw_args)
        
        # end dictionary-related code
        
        
        # 6/10 : https://www.kite.com/python/answers/how-to-print-a-number-in-scientific-notation-in-python
        
        list_args = []
        
        for key in which_args:
            value = these_args[key]
            value = "{:.2e}".format(value) if not isinstance(value, str) else value
            list_args.append(value)
            
        # end scientific notation code
            
        rep_string = '/'.join(list_args)
        
        # 6/9 : Datetime now: https://www.programiz.com/python-programming/datetime/current-datetime
        
        time_string = str(datetime.now()).replace(' ', 'T').replace(':','').replace('-','')

        # end datetime code
        
        # You will use this to generate the folder, then the name of the experiment.
        return rep_string, time_string
