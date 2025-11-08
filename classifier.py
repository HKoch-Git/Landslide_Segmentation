
############################################
## Files were changed from original!     ###
############################################


# 6/8: imports from
# https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_image_classifier.py

# Details on how logging works, etc. are here 6/23/21: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html

# begin imports taken from lit_image_classifier code
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

import torch.nn as nn # added this line

from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
# removed one import with MNIST

# end imports taken from the lit_image_classifier code

from parsers import train_parser, eval_parser
import segmentation_models_pytorch as smp
from models import unet
from models import nestedunet


import os
from os.path import join, exists



############################################
## Files were changed from original!     ###
############################################
    
# 6/8: Entire class code taken and adapted from 
# https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_image_classifier.py

# Changes made to the file: Logic inside most of the functions especially where specific to model declaration/usage (such as use of lr scheduler), except for template-like logic (things like how to forward things through a model, then compute loss, etc.)
# Including some arguments to functions.
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
class SinkholeSegmenter(pl.LightningModule):
    
    def __init__(self, input_type, model_type, norm_type, gpus, learning_rate, weight_decay, lr_decay_every, lr_decay, feature_weight,encoder):
        
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_decay_every = lr_decay_every
        self.lr_decay = lr_decay

        self.val_loss_list = []
        
        
        # 6/9 : save_hyperparameters line from: https://pytorch-lightning.readthedocs.io/en/stable/common/hyperparameters.html
        self.save_hyperparameters()
        # end save hyperparams
        

        self.criterion_weight = torch.tensor([1.0 / feature_weight, 1.0]).cuda()
        self.criterion = nn.CrossEntropyLoss(weight=self.criterion_weight, reduction='sum')
        
        # 6/8: https://stackoverflow.com/questions/11479816/what-is-the-python-equivalent-for-a-case-switch-statement
        # Dictionary substitution for case statement
        
        input_channel_cases = {
            'dem' : 1,
            'shaded' : 3,
            'naip' : 4,
            'dem_ddxy' : 2,
            'dem_dxy_pre' : 1,
            'spp': 3,
        }
        
        # end dictionary substitution
        
        # 6/9: debugging help (not specific code citation) https://stackoverflow.com/questions/18053500/typeerror-not-all-arguments-converted-during-string-formatting-python
        
        self.input_channels = input_channel_cases[input_type]

        model_params = {'in_channels': self.input_channels, 'classes': 2, 'encoder_name': encoder}
        
        self.model = {
            #'unet' : unet.Unet(in_channels=self.input_channels, out_channels=2, feature_reduction=4, norm_type=norm_type),
            #'nestedunet': nestedunet.NestedUNet (num_classes=2, input_channels=self.input_channels, deep_supervision=False),
            'unet': smp.Unet,
            'unetpp': smp.UnetPlusPlus,
            'FPN': smp.FPN,
            'dl3p': smp.DeepLabV3Plus,
            'manet': smp.MAnet,
            'lnet': smp.Linknet
        }[model_type](**model_params)
        
        if gpus >= 1:
            self.model.to('cuda:0')

    # 6/15 : Reference for general structure of forward vs. training/val/test separation in this file.
    # https://pytorch-lightning.readthedocs.io/_/downloads/en/stable/pdf/
        
    def forward(self, this_data):
        return self.model(this_data)
    
    def training_step(self, batch, batch_idx):
        
        this_data, this_label = batch
        
        y_hat = self(this_data)
        
        loss = self.criterion(y_hat, this_label)
        self.log('train_loss', loss, on_epoch = True, on_step = False)
        
        # 7/20/21 Update : seems like lr is called automatically.
        
        return loss
    
 
    def validation_step(self, batch, batch_idx):
        
        # 6/9: torch.no_grad is automatically called by pl (not code taken, just a reference)
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2171
        
        this_data, this_label = batch
        
        y_hat = self(this_data)

        loss = self.criterion(y_hat, this_label)
        
        self.log('val_loss', loss, on_epoch = True, on_step = False)
        
        return loss


    def configure_optimizers(self):
        
        this_optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        this_scheduler = torch.optim.lr_scheduler.StepLR(this_optim, step_size=self.lr_decay_every, gamma=self.lr_decay)
        
        return {
            'optimizer': this_optim,
            'lr_scheduler': this_scheduler,
        }
    
    # end the code from the class taken from the template code