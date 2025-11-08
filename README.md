# Geologic Feature Segmentation
A code to use image segmentation models to extract geologic features from elevation and other imaging data, specifically landslides.

## Description
This code currently supports a number of approaches, including: 
  * U-Net 
  * U-Net++
  * MA-Net
  * FPN
  * Linknet
  * PSP-Net
  * DeepLabV3

and six input options:
  * DEM
  * Slope
  * DEM gradient
  * SPP
  
## Getting Started 
Creating an environment

Clone the repository and run the following command:
```
conda env create -f ptl_smp.yml
```
## Training

To train the current best-performing model:
```
python train.py --model_type <model you are using,e.g. unet or nestedunet>  --input_type <the type of data you are using, e.g. dem_ddxy>
```
Further options are described in parsers/train_parser.py 

Edit base_dir and filenames of the 'ky' block of filenames_config.py to match your own data.

## Evaluation
```
python eval.py --phase test --checkpoint_path <your checkpoint path> 
```
Make sure you have configured your data directory with the training data from the checkpoint that you are using, as specified in filenames_config.py.  

## Inference
```
python inference_main.py  --input_type <the type of data you are using, e.g. dem_ddxy>  --checkpoint_path <path to your checkpoint>
```

**Note: You need to run evaluation first before running inference because evaluation outputs model configuration and the best threshod.**

To add a dataset for inference, go to filenames_config.py in the repository root and edit base_dir and filenames of the 'ky_inference' block for your own inference data.
  

  

# Attributions
This code is developed upon the code written by Rafique et al (https://github.com/mvrl/sink-seg/). A paper associated with the code can be found at Koch et al. (2026) 

Segmentation approach and encoders use backbone from Segmentation Models, Iakubovskii (2019; https://github.com/4uiiurz1/pytorch-nested-unet)

Training data available through https://zenodo.org/records/17559655
