
import torch
import torch.nn as nn
import numpy as np
import os
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import filenames_config
from data_utils import data_types_funcs, derivs


def load_dem(path_dict):
    
    dem_name = os.path.join(path_dict['base_dir'], path_dict['dem'])
    dem_full = Image.open(dem_name)
    
    # image size is the transpose of array size
    assert isinstance(dem_full, PIL.TiffImagePlugin.TiffImageFile), "DEM must be a PIL Image for dimensions to work out."
    # If its size is a single integer, then it's an array,
    # and the dimensions computations later won't work.
    
    return dem_full
       
    
def load_data(dataset_name, used_keys):
    """
    Outputs are PIL Images (except dem_dx, dem_dy)
    """

    available_keys = list(data_types_funcs.available_data_keys(dataset_name).keys())
    dataset_filenames = filenames_config.dataset_filenames[dataset_name]
    
    data_dir = dataset_filenames['base_dir']
    # load shaded relief
    
    used_data = {}
    
    if 'shaded' in used_keys:
 
        image_name = os.path.join(data_dir, dataset_filenames['shaded'])
        image_full = Image.open(image_name)
        used_data['shaded'] = image_full

    # This allows us to use all the operations provided in torchvision

    if 'dem' in used_keys:
    
        used_data['dem'] = load_dem(dataset_filenames)
    
    if 'dem_dx' in available_keys:
        assert 'dem_dy' in available_keys, "dem_dx and dem_dy data should always be available together."
        
    # Commented out version may be helpful for fusion version
    if 'dem_ddxy' in used_keys: #  #any(s in used_keys for s in ['dem_ddxy', 'dem_dx', 'dem_dy']):
        
        # Automatically compute dem_dx, dem_dy if it's not available.
        
        if 'dem_dx' not in available_keys:
            
            assert 'dem' in available_keys, "Need to either provide precomputed dem_dx.npy and dem_dy.npy or provide dem data in the filenames_config file."
            
            dem_dx, dem_dy = derivs.process_dem_ddxy(load_dem(dataset_filenames))
                
        else:
            dem_dx = np.load(os.path.join(data_dir, dataset_filenames['dem_dx']))
            dem_dy = np.load(os.path.join(data_dir, dataset_filenames['dem_dy']))
            
        used_data['dem_dx'] = dem_dx
        used_data['dem_dy'] = dem_dy 

    # Native, pre-computed derivatives
    
    if 'dem_dxy_pre' in used_keys:
        dem_dxy_pre_name = os.path.join(data_dir, dataset_filenames['dem_dxy_pre'])
        dem_dxy_pre = Image.open(dem_dxy_pre_name)
        used_data['dem_dxy_pre'] = dem_dxy_pre
    
    if 'naip' in used_keys:
        NAIP_name = os.path.join(data_dir,dataset_filenames['naip'])
        NAIP_full = Image.open(NAIP_name)
        used_data['naip'] = NAIP_full
        
    if 'spp' in used_keys:
        
        if 'dem_dxy_pre' not in available_keys:
             assert 'dem_dxy_pre' in available_keys, "Need to provide slope data in the filenames_config file."
            
        if 'plan_curv' not in available_keys:    
            assert 'plan_curv' in available_keys, "Need to provide plan curvature data in the filenames_config file."     
             
        if 'prof_curv' not in available_keys:    
            assert 'prof_curv' in available_keys, "Need to provide profile curvature data in the filenames_config file."        

        slope_name =  os.path.join(data_dir,dataset_filenames['dem_dxy_pre']) 
        used_data['slope']=np.array(Image.open(slope_name))
        plan_curv_name = os.path.join(data_dir,dataset_filenames['plan_curv']) 
        used_data['plan_curv']=np.array(Image.open(plan_curv_name))
        prof_curv_name = os.path.join(data_dir,dataset_filenames['prof_curv'])
        used_data['prof_curv']=np.array(Image.open(prof_curv_name))
        
    # Note that, when I was processing the entire dataset, i.e. without splits (for code development purposes only to see if statistics matched expected values), this will technically crop off a pixel for the NAIP image.
    # For KY data NAIP is shape (14268, 18851)
    # For the rest of the data types the KY data is (14267, 18850)
    # But this is likely desired behavior overall, because the fusion inputs will need to be the same size, so it makes sense to cut them to be the smaller pixel size.
    
    if 'labels' in dataset_filenames:
        
        label_name = os.path.join(data_dir, dataset_filenames['labels'])
        label_full = Image.open(label_name)
        used_data['labels'] = label_full

    return used_data


def compute_normalization_stats(phase_data, used_data_keys, norm_on = 'train'): 
    
    if 'dem_ddxy' in used_data_keys:
        used_data_keys += ['dem_dx', 'dem_dy']

    if 'spp' in used_data_keys:
        used_data_keys += ['slope','plan_curv','prof_curv']
    
    # Need to calculate train-based statistics for normalization.
    
    # This is called 3d but really NAIP is 4d. However "3d" applies to both 3d/4d inputs -- the code is still the same.
    norm_3d_mean = lambda arr : np.mean(arr, axis=(0,1))
    norm_3d_stdev = lambda arr : np.std(arr, axis=(0,1))
    
    set_1d = (np.mean, np.std, np.min, np.max)
    set_3d = (norm_3d_mean, norm_3d_stdev, np.min, np.max)
    
    # Above: set_3d still has global min/max because original code
    # implies global normalization for 0-to-1.
    # The channel-wise norm seems to be only used for unit gaussian normalization.
    
    norm_funcs = {
        'shaded' : set_3d,
        'dem' : set_1d,
        'dem_dx' : set_1d,
        'dem_dy' : set_1d,
        'dem_dxy_pre' : set_1d,
        'naip' : set_3d,
        'slope': set_1d,
        'plan_curv': set_1d,
        'prof_curv': set_1d
    }
    
    norm_stats = {}
    smallest_channel = lambda arr : min(arr.shape) == arr.shape[2]
        
    for used_key in used_data_keys:
        
        print(used_key)
        
        if used_key == 'dem_ddxy': # Replace dem_derivative with its two components.
            continue

        if used_key == 'spp':
            continue
        # You actually don't need to compute this unless 0-to-1 or unit_gaussian is specified for that input
        
        mean_func, stdev_func, min_func, max_func = norm_funcs[used_key]
        norm_stats[used_key] = {}
        
        raw_data = phase_data[norm_on][used_key]
        
        # If the data is not yet a numpy array, then make it so
        this_data = np.array(raw_data) if used_key not in {'dem_dx', 'dem_dy'} else raw_data 

        uses_3d = {'shaded', 'naip'}

        if used_key in uses_3d:
            print('for train shape', used_key, this_data.shape)
            assert smallest_channel(this_data), f"The non-hw channel is not last in the 3d input for {normed_name}"
        
        for stat, stat_func in zip(['mean', 'stdev', 'min', 'max'], norm_funcs[used_key]):
            print(f'Now getting stats for data type: {used_key}')
            norm_stats[used_key][stat] = stat_func(this_data)
            
    return norm_stats
    
    

def get_normalized_data_and_stats(args, dataset_name = 'ky', norm_on = 'train'): 
    """
    norm_on should always be equal to 'train'.
        the only reason to set it otherwise is to 'all_debug', and that was for checks
        for consistency with old code.
    """
    
    phases = ['train', 'val', 'test']
    
    dataset_filenames = filenames_config.dataset_filenames[dataset_name]
    
    all_input_types = ['dem', 'shaded', 'naip', 'dem_ddxy', 'dem_dxy_pre']
    
    # You will want to use available_data_keys for fusion work.
    used_data_keys = [args['input_type']] # list(data_types_funcs.available_data_keys(dataset_name).keys())
    
    if 'dem_ddxy' in used_data_keys:
        used_data_keys += ['dem_dx', 'dem_dy']
    
    # This is to maintain compatibility with selective loading in fusion code,
    # but focusing on current code for now.
    
    used_data = load_data(dataset_name, used_data_keys)
    
    if args['input_type'] == 'dem_ddxy':
        assert isinstance(used_data['dem_dx'], np.ndarray)
        reference_img = Image.fromarray(used_data['dem_dx'])
    elif args['input_type'] == 'spp':
        reference_img = Image.fromarray(used_data['slope'])
    else:
        reference_img = used_data[used_data_keys[0]]
        if (args['input_type'] == 'naip'):
            print('Please see Issue 27 on Github for notes on possibly unexpected behavior with NAIP vs. other dimension issues.')
        
    reference_width, reference_height = reference_img.size
    
    # width_ratio = args['train_width_percent']
    # height_ratio = args ['train_height_percent']
    # val_test_split = args ['val_test_split']
    
    # if width_ratio > 0.95:
    #     if height_ratio > 0.95:
    #         height_ratio = 0.75
    #     train_width=reference_width
    #     train_height=int(height_ratio*reference_height)
        
    #     val_orig_col = 0
    #     val_orig_row = train_height
    #     val_width=int(val_test_split*reference_width)
    #     val_height=reference_height

    #     test_orig_col=val_width
    #     test_orig_row = train_height
    #     test_width = reference_width
    #     test_height =reference_height
    
    # else: #when width_ratio is specified (<0.95), ignore train_height_percent

    #     train_width=int(width_ratio*reference_width)
    #     train_height=reference_height
        
    #     val_orig_col=train_width
    #     val_orig_row=0
    #     val_width=reference_width
    #     val_height=int(val_test_split*reference_height)
        
    #     test_orig_col=train_width
    #     test_orig_row=val_height
    #     test_width = reference_width
    #     test_height =reference_height

    side = args['train_split_orientation']
    train_ratio = args['train_split']
    val_test_ratio = args['val_test_split']

    if side in ['left', 'right']:
        p = int(reference_width * train_ratio)
        q = int(reference_height * val_test_ratio)

        if side == 'left':
            train_orig_row, train_orig_col, train_height, train_width = 0, 0, reference_height, p
            val_orig_row, val_orig_col, val_height, val_width = reference_height - q, p, reference_height, reference_width
            test_orig_row, test_orig_col, test_height, test_width = 0, p, reference_height - q, reference_width
        else: # side == 'right'
            train_orig_row, train_orig_col, train_height, train_width = 0, reference_width - p, reference_height, reference_width
            val_orig_row, val_orig_col, val_height, val_width = 0, 0, q, reference_width - p
            test_orig_row, test_orig_col, test_height, test_width = q, 0, reference_height, reference_width - p
    else: # side in ['top', 'bottom']
        p = int(reference_height * train_ratio)
        q = int(reference_width * val_test_ratio)

        if side == 'top':
            train_orig_row, train_orig_col, train_height, train_width = 0, 0, p, reference_width
            val_orig_row, val_orig_col, val_height, val_width = p, 0, reference_height, q
            test_orig_row, test_orig_col, test_height, test_width = p, q, reference_height, reference_width
        else: # side == 'bottom'
            train_orig_row, train_orig_col, train_height, train_width = reference_height - p, 0, reference_height, reference_width
            val_orig_row, val_orig_col, val_height, val_width = 0, reference_width - q, reference_height - p, reference_width
            test_orig_row, test_orig_col, test_height, test_width = 0, 0, reference_height - p, reference_width - q

    print(f'training width = {100*train_width/reference_width}%, width = {train_width}')
    print(f'training height = {100*train_height/reference_height}%, height = {train_height}')
    
    print(f'orientation: {side}')
    print(f'train region (row,col): {(train_orig_row,train_orig_col)} to {(train_height, train_width)}')
    print(f'val region (row,col): {(val_orig_row,val_orig_col)} to {(val_height, val_width)}')
    print(f'test region (row,col): {(test_orig_row,test_orig_col)} to {(test_height, test_width)}')
    
    ## Need to refactor this to only include relevant data dynamically.
    
    phase_data = { k_phase : {}  for k_phase in phases}
    phase_labels = {}

    # shaded relief
    crop_names = {'shaded', 'dem', 'dem_dxy_pre', 'naip'} & set(used_data.keys())
    
    ##phase_crops = {
    ##    'train' : (0,0,train_width,train_val_height),
    ##    'val' : (train_width,0,reference_width, train_val_height),
    ##    'test' : (0,train_val_height,reference_width,reference_height),
    ##}
    
    phase_crops = {
        'train' : (train_orig_col,train_orig_row,train_width,train_height),
        'val' : (val_orig_col,val_orig_row, val_width, val_height),
        'test' : (test_orig_col,test_orig_row, test_width, test_height),
    }

    
    for name in crop_names: # Data that uses crops (i.e. is not dem_dx or dem_dy)
        data = used_data[name]
        for phase, phase_crop in phase_crops.items():
            
            print(f"now cropping for phase: {phase}, data type: {name}")
            phase_data[phase][name] = data.crop(phase_crop)
            phase_data[phase][name].load() 
    
    # Crop labels and dem_ref separately to avoid introducing it into phase_data dictionary
    
    for phase, phase_crop in phase_crops.items():
        print(f"now cropping for phase: {phase}, data type: labels")
        phase_labels[phase] = used_data['labels'].crop(phase_crop)
        phase_labels[phase].load()
    
    
    # DEM derivative crops, separate
    
    if 'dem_ddxy' in used_data_keys:
        
        dem_dx = used_data['dem_dx']
        dem_dy = used_data['dem_dy']
        
        ##phase_data['train']['dem_dx']= dem_dx[:train_val_height, :train_width]
        ##phase_data['train']['dem_dy'] = dem_dy[:train_val_height, :train_width]

        ##phase_data['val']['dem_dx'] = dem_dx[0:train_val_height, train_width:]
        ##phase_data['val']['dem_dy'] = dem_dy[0:train_val_height, train_width:]

        ##phase_data['test']['dem_dx'] = dem_dx[train_val_height:, :]
        ##phase_data['test']['dem_dy'] = dem_dy[train_val_height:, :]

        phase_data['train']['dem_dx']= dem_dx[train_orig_row:train_height, train_orig_col:train_width]
        phase_data['train']['dem_dy'] = dem_dy[train_orig_row:train_height, train_orig_col:train_width]

        phase_data['val']['dem_dx'] = dem_dx[val_orig_row:val_height, val_orig_col:val_width]
        phase_data['val']['dem_dy'] = dem_dy[val_orig_row:val_height, val_orig_col:val_width]

        phase_data['test']['dem_dx'] = dem_dx[test_orig_row:test_height, test_orig_col:test_width]
        phase_data['test']['dem_dy'] = dem_dy[test_orig_row:test_height, test_orig_col:test_width]


    if 'spp' in used_data_keys:
        #slope = np.array(used_data['slope'])
        #plan_curv= np.array(used_data['plan_curv'])
        #prof_curv = np.array(used_data['prof_curv'])

        slope = used_data['slope']
        plan_curv= used_data['plan_curv']
        prof_curv = used_data['prof_curv']

        phase_data['train']['slope']= slope[train_orig_row:train_height, train_orig_col:train_width]
        phase_data['train']['plan_curv'] = plan_curv[train_orig_row:train_height, train_orig_col:train_width]
        phase_data['train']['prof_curv'] = prof_curv[train_orig_row:train_height, train_orig_col:train_width]

        phase_data['val']['slope'] = slope[val_orig_row:val_height, val_orig_col:val_width]
        phase_data['val']['plan_curv'] = plan_curv[val_orig_row:val_height, val_orig_col:val_width]
        phase_data['val']['prof_curv'] = prof_curv[val_orig_row:val_height, val_orig_col:val_width]

        phase_data['test']['slope'] = slope[test_orig_row:test_height, test_orig_col:test_width]
        phase_data['test']['plan_curv'] = plan_curv[test_orig_row:test_height, test_orig_col:test_width]   
        phase_data['test']['prof_curv'] = prof_curv[test_orig_row:test_height, test_orig_col:test_width]   


    norm_stats = compute_normalization_stats(phase_data, used_data_keys, norm_on)
     
    return phase_data, phase_labels, norm_stats
