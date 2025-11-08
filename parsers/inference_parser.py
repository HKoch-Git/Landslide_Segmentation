
# 6/8: Not citing for any substantial/unique code here, but just for general organization idea of segmenting a parser into a separate function
# https://github.com/NVlabs/SPADE/blob/master/options/base_options.py (for the code/class definition)


import argparse

def get_inference_parser():
    
    parser = argparse.ArgumentParser()
    
    # 7/27/21: https://docs.python.org/3/library/argparse.html
    # Syntax reference
    
    parser.add_argument('--inference_dataset_name', type = str, default = 'ky_inference', help = "A short name for your dataset.")
    
    # Below: Until the codebase is converted to fusion, accept only a single input even if user specifies multiple.
    # The current codebase is setup so it should hopefully be relatively easy to extend to fusion/multiple inputs,
    # But changes will have to be made.
    
    parser.add_argument('--input_type', type = str, default = 'dem_ddxy', help = "Which input to use. This codebase doesn't support fusion yet.")
    
    parser.add_argument('--output_dir', type = str, nargs = "?", default='./inference_records', help = "Where to save the inference result.")
    
    parser.add_argument('--checkpoint_path', nargs = "?", default = 'unspecified', type = str, help = "The absolute path to your Pytorch Lightning checkpoint. Should be in a folder marked 'checkpoints' relative to your experiment root folder.")  
    
    # end cite
    
    return parser 


# end cite
