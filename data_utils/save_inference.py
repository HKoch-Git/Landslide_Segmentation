# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:47:20 2022

@author: jzh226
"""
import os
import rasterio
import filenames_config

from os.path import join, exists
from PIL import Image

##save inference result with geo information
def save_inference_results(output_dir, area_for_save, used_keys,infer_name,display_name):
    
    #get input image path & name
    dataset_name= 'ky_inference'
    dataset_filenames = filenames_config.dataset_filenames[dataset_name]
    data_dir = dataset_filenames['base_dir']
    
   
    if 'shaded' in used_keys:
        image_name = os.path.join(data_dir, dataset_filenames['shaded'])

    if 'dem' in used_keys:
        image_name = os.path.join(data_dir, dataset_filenames['dem']) 

    if 'dem_ddxy' in used_keys:
        image_name = os.path.join(data_dir, dataset_filenames['dem'])

    if 'naip' in used_keys:
        image_name = os.path.join(data_dir, dataset_filenames['naip'])

    if 'spp' in used_keys:
        image_name = os.path.join(data_dir, dataset_filenames['dem_dxy_pre'])
   
    ##save inference results with geo-reference information from input image
    tiff=rasterio.open(image_name)
    outName=join(output_dir, infer_name)
        
    newdataset=rasterio.open(
       outName,
       'w',
       drive='GTiff',
       height=tiff.height,
       width=tiff.width,
       count=1,
       dtype=area_for_save.dtype,
       crs=tiff.crs,
       transform=tiff.transform
       )
   
    newdataset.write(area_for_save,1)
    newdataset.close()
    
    #save the same result as a png file
    display_inference_path = join(output_dir, display_name)
    display_inference = Image.fromarray((area_for_save * 255.0).astype('uint8'))
    display_inference.save(display_inference_path)
   
    print(f'Saved inference results to {outName}.')
    print(f'For visible inference results (not the raw prediction), please see {display_inference_path}')    
    
    return
    
