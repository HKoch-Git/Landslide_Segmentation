
# ID of the placeholder checkpoint for use in inference.
# This is not a checkpoint intended for use with final results. 

#print("Warning: If inferencing with default checkpoint, currently using a strictly placeholder checkpoint! This was trained just for use as a placeholder shouldn't be used for results.")

gdrive_id = '12RYBcX35z_GDciCv2VtLVttAKKZ6c-Y9'

dataset_filenames = {
    
    'ky': {

        # landslide inputs
         'base_dir' : '../data',
         #shaded' : ,
         'dem' : 'DEM_5ft_cut.tif',
         'dem_dxy_pre' : 'Slope_5ft_cut.tif',
         #'naip' :  ,
         'plan_curv':  'PlanCurv_5ft_cut.tif',
         'prof_curv':  'ProfCurv_5ft_cut.tif',
         'labels' : 'BinaryLandslidesCut.tif'      
    },

    
    # Not integrating non-dem deriv data types or auto dem deriv compute for now
    'ky_inference' : {

        #landslide inference inputs,
        # allowing for inference with dem_ddxy (use DEM as input) or
        # spp (use slope + plan Curvature + Profile Curvature as inputs)
        'base_dir' :  '../inference',
        'dem' : 'Cutshin.tif',
        'dem_dxy_pre' : 'CutshinSlope.tif', 
        'plan_curv':  'CutshinPlanCurv.tif', 
        'prof_curv':  'CutshinProfCurv.tif', 
        #'shaded' : ,
        #'naip': 
    },

# Below is for development only, to see if it works with a different training dataset
    'ky_limited': {
        'base_dir' : '/u/vul-d1/data/sinkhole/v1',
        'shaded' : 'ShadedRelief_Raster.tif',
        'dem_dx' : 'dem_dx.npy',
        'dem_dy' : 'dem_dy.npy',
        'naip' : 'Ky_NAIP_2018_5FT.tif',
        'labels' : 'SinkholeBinaryRaster.tif'

    }
     
}