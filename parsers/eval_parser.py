# 6/8: Organization, imports, and all initialization code/class structure taken from:
# https://github.com/NVlabs/SPADE/blob/master/options/base_options.py (for the code/class definition)
# https://github.com/NVlabs/SPADE/blob/master/train.py (for the hierarchical structure)

# Modifications made to code: Almost all logic in the file, except for the logic/code described above. Lines/function organization was taken in pieces from the original file.

import argparse
from parsers import gen_parser

import os

class EvalParser(gen_parser.GenParser):
    
    def initialize(self, parser):
        
        self.initialized = False
        
        # 6/14: https://stackoverflow.com/questions/24180527/argparse-required-arguments-listed-under-optional-arguments
        
        parser.add_argument('--phase', type = str, required = True, help = 'Either "val" or "test".')
        parser.add_argument('--checkpoint_path', type = str, required = True, help = 'Which model checkpoint to use. \{best, end\}')
        
        self.initialized = True
        
        return parser 
    
    
# end structure/taken code from the repository
