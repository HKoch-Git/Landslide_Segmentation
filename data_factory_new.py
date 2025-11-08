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

from data_utils import get_raw_stats_data, derivs, data_types_funcs

class dataset_sinkhole(Dataset):
    """
    This assumes one input type for now
        (no fusion)
    """
    
    def __init__(self, args, phase, this_phase_data, this_phase_labels, norm_stats, is_eval_mode, is_inference):
        
        self.args = args
        self.phase = phase
        
        # Note that I didn't check if dim 0 or dim 1 conventionally corresponds to w or h.
        # I only take in one argument for cutout_tuple so it's always square in this codebase.

        self.cutout_w, self.cutout_h = (args['cutout_size'], args['cutout_size'])
        
        self.norm_stats = norm_stats
        self.eval_pad = is_eval_mode
        self.is_inference = is_inference
        pad_w = 0
        pad_h = 0
        if (args['input_type'] == 'dem_ddxy'):
            tensor_dx = torch.from_numpy(this_phase_data['dem_dx'])
            tensor_dy = torch.from_numpy(this_phase_data['dem_dy'])
            self.original_shape = list(tensor_dx.size())  
            
            if (tensor_dx.size()[0] % self.cutout_h != 0):
                num_height = tensor_dx.size()[0] // self.cutout_h + 1
                pad_h = num_height * self.cutout_h - tensor_dx.size()[0]

            if (tensor_dx.size()[1] % self.cutout_w != 0):
                num_width = tensor_dx.size()[1] // self.cutout_w + 1
                pad_w = num_width * self.cutout_w - tensor_dx.size()[1]

            this_phase_data['dem_dx'] = np.pad (this_phase_data['dem_dx'], ([0, pad_h], [0,pad_w]), 'reflect')
            this_phase_data['dem_dy'] = np.pad (this_phase_data['dem_dy'], ([0, pad_h], [0,pad_w]), 'reflect')

        elif (args['input_type'] == 'spp'):
            tensor_slope = torch.from_numpy(this_phase_data['slope'])
            self.original_shape = list(tensor_slope.size())

            if (tensor_slope.size()[0] % self.cutout_h != 0):
                num_height = tensor_slope.size()[0] // self.cutout_h + 1
                pad_h = num_height * self.cutout_h - tensor_slope.size()[0]

            if (tensor_slope.size()[1] % self.cutout_w != 0):
                num_width = tensor_slope.size()[1] // self.cutout_w + 1
                pad_w = num_width * self.cutout_w - tensor_slope.size()[1]

            this_phase_data['slope'] = np.pad (this_phase_data['slope'], ([0, pad_h], [0,pad_w]), 'reflect')
            this_phase_data['plan_curv'] = np.pad (this_phase_data['plan_curv'], ([0, pad_h], [0,pad_w]), 'reflect')
            this_phase_data['prof_curv'] = np.pad (this_phase_data['prof_curv'], ([0, pad_h], [0,pad_w]), 'reflect')

        else:
            self.original_shape = (this_phase_data.size[0],this_phase_data.size[1])

            if(this_phase_data.size[0] % self.cutout_h != 0):
                num_height = this_phase_data.size[0] // self.cutout_h + 1
                pad_h = num_height * self.cutout_h - this_phase_data.size[0]

            if (this_phase_data.size[1] % self.cutout_w != 0):
                num_width = this_phase_data.size[1] // self.cutout_w + 1
                pad_w = num_width * self.cutout_w - this_phase_data.size[1]   
        
            this_phase_data = transforms.Pad(padding=(0,0,pad_w,pad_h), padding_mode='reflect')(this_phase_data)
        self.phase_data = this_phase_data
        #self.reference = this_phase_data if (args['input_type'] != 'dem_ddxy') else Image.fromarray(this_phase_data['dem_dx'])
        if (args['input_type'] == 'dem_ddxy'):
            self.reference = Image.fromarray(this_phase_data['dem_dx'])
        elif  (args['input_type'] == 'spp'):
            self.reference = Image.fromarray(this_phase_data['slope'])
        else:
            self.reference = this_phase_data

        # Inference requires a placeholder because labels are not guaranteed to be available,
        # but for compatibility with general evaluation code that uses dataloaders,
        # cannot use a placeholder of None.

        self.np_shape = (self.reference.size[1], self.reference.size[0])
        self.phase_labels = this_phase_labels if not self.is_inference else Image.fromarray(np.ones(self.np_shape) * -1) 
        
    def get_pred_area_dims_rc(self):
        """
        Number of rows and columns that fit into the inference area
        """
        
        w, h = self.reference.size
        num_in_w = int(w / self.cutout_w)
        num_in_h = int(h / self.cutout_h)
        
        return num_in_w, num_in_h
    
    def get_pred_area_dims_pixels(self):
        """
        Size of the inference area
            (not the same as the inference area because
                it's possible you can't fit an extra cutout size in the last row/column)
        """
        
        num_w, num_h = self.get_pred_area_dims_rc()
        size_x, size_y = self.cutout_w * num_w , self.cutout_h * num_h

        return size_x, size_y
        
    def __len__(self):
        
        num_in_w, num_in_h = self.get_pred_area_dims_rc()
        
        num_cutouts = num_in_w * num_in_h
        
        return num_cutouts
    
    ## Below functions used for inference reconstruction
    
        
    def get_rc_relative_to_pad(self, idx):
        """
        Finds the position of corners of the center of the crop (no padding)
            if the left/upper right corner is (0,0)
        """
        
        orig_left, orig_upper, orig_right, orig_lower = self.get_rc_from_index_no_pad(idx)
        new_left, new_upper, new_right, new_lower = self.get_rc_from_index(idx)
        # Above: With padding, because rc functions are only called if padding-related.
        
        left_rel_pad_corner = orig_left - new_left
        upper_rel_pad_corner = orig_upper - new_upper
        
        # The original lower corner relative to the padding corner (which is farther up/left)
        right_rel_pad_corner = orig_right - new_left
        lower_rel_pad_corner = orig_lower - new_upper 
        

        rel_corner = (
            left_rel_pad_corner,
            upper_rel_pad_corner,
            right_rel_pad_corner,
            lower_rel_pad_corner,
        )
        
        assert all(c >= 0 for c in rel_corner)
        
        return rel_corner
    
    def limit_dims_range(self, a_left, a_upper, a_right, a_lower):
        
        ref_w, ref_h = self.reference.size
        
        left = max(a_left, 0)
        upper = max(a_upper, 0)
        right = min(a_right, ref_w)
        lower = min(a_lower, ref_h)
        
        return left, upper, right, lower
        
        
    def get_rc_from_index(self, idx):
        
        r_left, r_upper, r_right, r_lower = self.get_rc_from_index_no_pad(idx)
        
        # Enforce that no out-of-bounds to avoid PIL padding with zeros
        
        left, upper, right, lower = self.limit_dims_range(
            #change padding to 80 to accoumadate nestedunet
            r_left - 64,
            r_upper -  64,
            r_right + 64,
            r_lower + 64
            #r_left - 40,
            #r_upper - 40,
            #r_right + 40,
            #r_lower + 40

         )

        return (left, upper, right, lower)
    
    def get_rc_from_index_no_pad(self, idx):
        
        num_columns = self.get_pred_area_dims_rc()[0]
            
        row, col = divmod(idx, num_columns)

        # The actual location of the eval cutout (not padding)
        anchor_left = col*self.cutout_w 
        anchor_upper = row*self.cutout_h
        
        # Left, upper, right, lower
        return self.limit_dims_range(anchor_left, anchor_upper, anchor_left + self.cutout_w, anchor_upper + self.cutout_h)

            
    def __getitem__(self, idx):
    
        #####################
        #### Cropping #######
        #####################
        
        is_dem_deriv = (self.args['input_type'] == 'dem_ddxy')
        is_spp = (self.args['input_type'] == 'spp')
        ref_w, ref_h = self.reference.size
        
        if is_dem_deriv:
            
            # Assumes you have already computed the proper dem_dx, dem_dy
            # For inference, you should add computations for this in the inference run script.
            
            dem_dx, dem_dy = self.phase_data['dem_dx'], self.phase_data['dem_dy']

        if is_spp:
            slope, plan_curv, prof_curv = self.phase_data['slope'], self.phase_data['plan_curv'], self.phase_data['prof_curv']
            
        if self.phase == 'train':
            
            # Note: Because this code expects square cutouts I did not check if 
            # transforms RandomCrop expects width-first or height-first for the output_size argument.
            
            cutout_tuple = (self.cutout_w, self.cutout_h) 
            i, j, h, w = transforms.RandomCrop.get_params(self.reference, output_size=cutout_tuple)
                
            if is_dem_deriv:
                
                assert dem_dx.shape == dem_dy.shape
                
                # Checked that dem_dx_ref and the other inputs are all the same shape (in unsplit form)
               
                cropped_dem_dx = data_types_funcs.crop_dem_deriv_offset(dem_dx, i, j, h, w)
                cropped_dem_dy = data_types_funcs.crop_dem_deriv_offset(dem_dy, i, j, h, w)

            elif is_spp:
                assert slope.shape == plan_curv.shape == plan_curv.shape
                cropped_slope = data_types_funcs.crop_dem_deriv_offset(slope, i, j, h, w)
                cropped_plan_curv = data_types_funcs.crop_dem_deriv_offset(plan_curv, i, j, h, w)
                cropped_prof_curv = data_types_funcs.crop_dem_deriv_offset(prof_curv, i, j, h, w)

            else:
                this_cropped_data = transforms_function.crop(self.phase_data, i, j, h, w)
            
            # Do this separately to avoid introducing labels into data-related loops.
            # Checked that labels is a PIL Image here
            this_cropped_labels = transforms_function.crop(self.phase_labels, i, j, h, w)
            
            # According to Pytorch docs i is associated with height
            assert (0 <= i < i+h <= ref_h) and (0 <= j < j+w <= ref_w), "Prevent possible unwanted PIL behavior if crop dimensions out of bounds"
        else: # val or test or inference           
            
            rc_crop_func = self.get_rc_from_index if self.eval_pad else self.get_rc_from_index_no_pad

            left, upper, right, lower = rc_crop_func(idx)
            
            # crop the dem derivatives
            
            if is_dem_deriv:
                cropped_dem_dx = data_types_funcs.crop_dem_deriv_corners(dem_dx, left, upper, right, lower)
                cropped_dem_dy = data_types_funcs.crop_dem_deriv_corners(dem_dy, left, upper, right, lower)

            elif is_spp:
                cropped_slope = data_types_funcs.crop_dem_deriv_corners(slope, left, upper, right, lower)
                cropped_plan_curv = data_types_funcs.crop_dem_deriv_corners(plan_curv, left, upper, right, lower)
                cropped_prof_curv = data_types_funcs.crop_dem_deriv_corners(prof_curv, left, upper, right, lower)
            else:
                this_cropped_data = self.phase_data.crop((left, upper, right, lower))
                this_cropped_data.load()

            # PIL will automatically pad the crop with zeros if the crop is out of bounds.
            # This will be prevented by?
            
            this_cropped_labels = self.phase_labels.crop((left, upper, right, lower))
            
            assert (0 <= left < right <= ref_w) and (0 <= upper < lower <= ref_h), "Prevent possible unwanted PIL behavior if crop dimensions out of bounds"
            
        #####################
        ##### Normalizing ###
        #####################     
        
        used_name = self.args['input_type']
        
        if is_dem_deriv:
            
            normed_dem_dx = data_types_funcs.normalize(cropped_dem_dx, self.args, self.norm_stats, used_name = 'dem_dx')
            normed_dem_dy = data_types_funcs.normalize(cropped_dem_dy, self.args, self.norm_stats, used_name = 'dem_dy')
            output = np.stack([normed_dem_dx, normed_dem_dy], axis = 0)

        elif is_spp:
            normed_slope = data_types_funcs.normalize(cropped_slope, self.args, self.norm_stats, used_name = 'slope')
            normed_plan_curv = data_types_funcs.normalize(cropped_plan_curv, self.args, self.norm_stats, used_name = 'plan_curv')
            normed_prof_curv = data_types_funcs.normalize(cropped_prof_curv, self.args, self.norm_stats, used_name = 'prof_curv')
            output = np.stack([normed_slope, normed_plan_curv,normed_prof_curv], axis = 0)

        else:
            output = data_types_funcs.normalize(this_cropped_data, self.args, self.norm_stats)
            
            # If there is no "depth" channel
            
            if len(output.shape) == 2:
                output = np.expand_dims(output, axis = 2) # Make depth channel -- note the next line will move it to the front.

            # I have a "depth" channel, but due to assert in get_raw_stats_data, it's in the back.
            # Move it to the front to prep for tensor expectations.
            output = np.transpose(output, (2, 0, 1))
            
            
        output = torch.from_numpy(output).float()
        
        output_labels = torch.from_numpy(np.array(this_cropped_labels)).long()
        
        # Note that if inference, output_labels is a placeholder array of -1 for compatibility with dataloaders.
        return (output, output_labels)


def get_inference_data(this_args, dataset_name = 'ky_inference', train_dataset_name = 'ky'):

    _, _, norm_stats = get_raw_stats_data.get_normalized_data_and_stats(args = this_args, dataset_name = this_args['train_dataset_name'])
    
    raw_inference_data = get_raw_stats_data.load_data(this_args['inference_dataset_name'], [this_args['input_type']])
    assert 'labels' not in raw_inference_data
     
    # Requires phase "dictionary" to be compatible with common code with val/test setup
    
    dset = dataset_sinkhole(this_args, phase = 'inference', this_phase_data = raw_inference_data, this_phase_labels = None, norm_stats = norm_stats, is_eval_mode = True, is_inference = True)
    datasets = {'inference' : dset}

    dataloaders = {'inference' : DataLoader(dset, batch_size=this_args['batch_size'], shuffle = False, num_workers=this_args['num_workers'])}
    return datasets, dataloaders

#Seed Worker Initialization (for DataLoader)
import random
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_all_data(this_args, is_eval_mode, dataset_name = 'ky'):
    
    norm_on = 'train'
    
    phases = ['train', 'val', 'test']
    
    input_name = this_args['input_type']
    
    data_func = get_raw_stats_data.get_normalized_data_and_stats
    
    all_phase_data, phase_labels, all_norm_stats = data_func(args = this_args, dataset_name = dataset_name, norm_on = norm_on)
    
    # For now, accept only one input.
    
    if input_name == 'dem_ddxy':
        phase_data = {phase : {this_type : all_phase_data[phase][this_type] for this_type in ['dem_dx', 'dem_dy']} for phase in phases}
        norm_stats = {k : all_norm_stats[k] for k in ['dem_dx', 'dem_dy']}

    elif input_name == 'spp':
        phase_data = {phase : {this_type : all_phase_data[phase][this_type] for this_type in ['slope', 'plan_curv','prof_curv']} for phase in phases}
        norm_stats = {k : all_norm_stats[k] for k in ['slope', 'plan_curv','prof_curv']}

    else:
        phase_data = { phase : all_phase_data[phase][input_name] for phase in phases }
        norm_stats = { input_name : all_norm_stats[input_name] }


    datasets = {
        phase : dataset_sinkhole(this_args, phase = phase, this_phase_data = phase_data[phase], this_phase_labels = phase_labels[phase], norm_stats = norm_stats, is_eval_mode = is_eval_mode, is_inference = False)
        for phase in phases
    }

    dataloaders = {
        phase : DataLoader(datasets[phase], batch_size=this_args['batch_size'], shuffle= (phase == 'train'), num_workers=this_args['num_workers'],worker_init_fn=seed_worker)
        for phase in phases
    }

    return datasets, dataloaders





    
