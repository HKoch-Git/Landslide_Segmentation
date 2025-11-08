import argparse
import os

from os.path import join, exists
import subprocess

from utils import path_utils
from eval_utils import load_eval_modules, inference_funcs
from data_factory_new import dataset_sinkhole
import shutil

import parsers
from parsers import inference_parser
from data_factory_new import dataset_sinkhole

import json

import glob

from PIL import Image

import filenames_config
import torch

from data_utils import save_inference

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if __name__ == '__main__':
    
    """
    Note: So far, the code is only really designed primarily for dem_ddxy inference,
        you may have to extend the codebase to support the other options.
    """
    
    parser = parsers.inference_parser.get_inference_parser()
    args = vars(parser.parse_args())

    output_dir = join(args['output_dir'], args['inference_dataset_name'])
    if not exists(output_dir):
        os.makedirs(output_dir)
    
    # Download and load the best checkpoint (use a clearly-marked placeholder for now)
    
    if args['checkpoint_path'] == 'unspecified':
        
        dst_path = './updated_checkpoint_with_threshold_with_args_dataset_name'
        
        from google_drive_downloader import GoogleDriveDownloader as gdd
        
        raw_dst_path = './placeholder_checkpoint.zip'
        
        gdd.download_file_from_google_drive(file_id=filenames_config.gdrive_id,
                                    dest_path=raw_dst_path,
                                    unzip=True)
        
        subprocess.call(f'unzip {raw_dst_path}', shell = True)

        
        is_ckpt = lambda path : '.ckpt' in path 
        checkpoint_path_list = list(filter(is_ckpt, glob.glob(join(dst_path, 'checkpoints/*'))))
        
        assert len(checkpoint_path_list) == 1, "Correctness of auto-download from Google Drive relies on this assumption (as does this codebase generally)."

        args['checkpoint_path'] = checkpoint_path_list[0]
        path = path_utils.get_experiment_folder(args['checkpoint_path'])
        
        if not exists(path):
            shutil.copytree(dst_path, path) 

    model, datasets, dataloaders = load_eval_modules.load_inference_model_data(args)
    # Run the inference
    best_threshold = load_eval_modules.load_best_threshold(args, dataloaders, model) 

    area_thresholded_pred, area_soft_pred = inference_funcs.gen_thresholded_inference(args, dataloaders, model, best_threshold)
    
    size_l, size_w = datasets['inference'].original_shape
    cropped_inference = area_thresholded_pred[0:size_l, 0:size_w]
    area_for_save = cropped_inference.detach().cpu().numpy()

    print('inference area', area_for_save.shape)
    
    save_inference.save_inference_results(output_dir, area_for_save, [args['input_type']],'inference_best_threshold.tif','display_inference_best_threshold.png')
    
    cropped_inference = area_soft_pred[0:size_l, 0:size_w]
    area_for_save = cropped_inference.detach().cpu().numpy()
    save_inference.save_inference_results(output_dir, area_for_save, [args['input_type']],'inference_soft.tif','display_inference_soft.png')    
