
from scipy.ndimage import convolve1d
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

def process_dem_ddxy(dem_data_img):
    
    dem_data = np.array(dem_data_img)
    
    sobel_h = [-0.5, 0, 0.5]
    sobel_v = [-0.5, 0, 0.5]
    x_derivative = convolve1d(dem_data, weights=sobel_h, axis=1)
    y_derivative = convolve1d(dem_data, weights=sobel_v, axis=0)
    
    return x_derivative, y_derivative

if __name__ == '__main__':
    
    file_path = '../../Landslide_image_data/Owsley5ftDEMCut.tif'
    
    dem_data = np.array(Image.open(file_path))
    
    xder, yder = process_dem_ddxy(dem_data)
    
    np.save ('Owsley_dem_dx.npy',xder)
    np.save ('Owsley_dem_dy.npy',yder)