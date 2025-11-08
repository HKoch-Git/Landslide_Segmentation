import numpy as np

from PIL import Image


import filenames_config

def available_data_keys(dataset_name):
    
    all_files = filenames_config.dataset_filenames[dataset_name]
    return { k : all_files[k] for k, v in all_files.items() if k != 'base_dir'}
    
def crop_dem_deriv_offset(dem_deriv, i, j, h, w):
    # i,j,h,w = upper, left, lower, right
    return dem_deriv[i:i+h, j:j+w]

def crop_dem_deriv_corners(dem_deriv, left, upper, right, lower):
    return dem_deriv[upper:lower, left:right]

def normalize(used_data, args, norm_stats, used_name = None):
    """
    Need to specify used_name = dem_dx, or dem_dy if one of these
    """
    
    if used_name is None:
        used_name = args['input_type']
    
    #query_name = used_name if used_name not in {'dem_dx', 'dem_dy'} else 'dem_ddxy'
    if used_name in {'dem_dx', 'dem_dy'}:
        query_name ='dem_ddxy'
    elif used_name in {'slope','plan_curv','prof_curv'}:
        query_name ='spp'
    else:
        query_name = used_name

    normalize_val = args[f'normalize_{query_name}']

    if normalize_val == '0_to_1':
        this_min = norm_stats[used_name]['min']
        this_max = norm_stats[used_name]['max']
        this_normed_data = (used_data - this_min) / (this_max - this_min + 1e-7)

    if normalize_val == 'unit_gaussian':
        this_mean = norm_stats[used_name]['mean']
        this_stdev = norm_stats[used_name]['stdev']
        this_normed_data = (used_data - this_mean) / this_stdev

    if normalize_val == 'instance':
        this_min_inst = np.min(used_data)
        this_max_inst = np.max(used_data)
        this_normed_data = (used_data - this_min_inst) / (this_max_inst - this_min_inst + 1e-7)

    if normalize_val == 'none':
        this_normed_data = np.array(used_data)

    return this_normed_data
