import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import  array_to_img, save_img, load_img

#------------ directories -------------

result_dir = './test-drhagis/'
output_images = './test-drhagis/output_images/cropped/'

#--------------------------------------
for result in os.listdir(result_dir):
     if result.endswith('.npy'):
         predicted_imgs = np.load(result_dir + result)
         predicted_imgs_ch = np.moveaxis(predicted_imgs, 1, 3)
         for i, image in enumerate(predicted_imgs_ch, 1):
             save_img(output_images + result[:-4] + f'_{i}.tif', image)