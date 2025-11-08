

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from os.path import join, exists

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from models.unet import Unet
from utils import path_utils

import json

import pprint

from eval_utils import eval_funcs


def gen_thresholded_inference(args, dataloaders, model, optimal_threshold):
    
    this_pred_area = gen_inference_over_area(args, dataloaders, model, optimal_threshold)
    thresholded_prediction = 1.0*(this_pred_area > optimal_threshold)
    return thresholded_prediction, this_pred_area


def gen_inference_over_area(args, dataloaders, model, optimal_threshold):
    
    this_data_loader = dataloaders[args['phase']]
    size_x, size_y = this_data_loader.dataset.get_pred_area_dims_pixels()
    pred_raw = torch.zeros(size_y, size_x)
    

    with torch.no_grad():
       
        for idx, (data, _) in enumerate(this_data_loader):
            
            predictions = model(data.cuda())
            predictions = torch.softmax(predictions, dim=1)
                
            startx, endx, starty, endy = eval_funcs.get_patch_idxs(idx, this_data_loader)


            predictions = predictions[:,:,starty:endy,startx:endx]
            left, upper, right, lower = this_data_loader.dataset.get_rc_from_index_no_pad(idx)

            # Limits above are swapped
            pred_raw[upper:lower, left:right] = predictions[:,1,:,:]
             
    return pred_raw




