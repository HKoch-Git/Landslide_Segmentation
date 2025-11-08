# 6/8: Organization, imports, and all initialization code/class structure taken from:
# https://github.com/NVlabs/SPADE/blob/master/options/base_options.py (for the code/class definition)
# https://github.com/NVlabs/SPADE/blob/master/train.py (for the hierarchical structure)

# Modifications made to code: Almost all logic in the file, except for the logic/code described above. Lines/function organization was taken in pieces from the original file.


import argparse

# Argparse import:
# 6/8: https://github.com/PyTorchLightning/deep-learning-project-template/blob/master/project/lit_image_classifier.py

from argparse import ArgumentParser 

# end argparse import taken

import os

class GenParser():
    
    # Model related arguments, including training/evaluation data specifications.
    
    def __init__(self):
        
        self.initialized = False
        self.args = None
    
    def initialize(self, parser):
        pass
        
        
    def parse(self):
        
        # 6/9 : The logic for gathering options and using the "parse" function is from here:
        # https://github.com/NVlabs/SPADE/blob/master/options/base_options.py
        # Modifications made to code: Rearranged the lines
        
        parser = ArgumentParser()
        parser = self.initialize(parser) # You should check what kind of initialize is used here?
        args = parser.parse_args() # These are the actual options.
        
        return args
    
# end structure/taken code from the repository