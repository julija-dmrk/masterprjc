import os

import cv2
import h5py
import numpy as np
from numpy import asarray
from PIL import Image
from pre_processing import get_fov
import matplotlib.pyplot as plt

### Path to the images #######
img_path = './all_data/train/laikinai/'
gth_path = './all_data/train/manual/'
msk_path = './all_data/train/masks/'

dim_image = './all_data/train/images/cropped/'
channels = 3


def crop_grt_msk (imgs_dir, groundTruth_dir, borderMasks_dir):
    files = os.listdir(imgs_dir)
    assert len(files) > 0
    img = cv2.imread(imgs_dir + files[0])
    for i in range(len(files)):
    # original
        print("original image: " + files[i])
    # corresponding ground truth
        groundTruth_name = files[i][0:5] + '_grt_' + files[i][10:13] + '.png'
        print("ground truth name: " + groundTruth_name)
        border_masks_name = files[i][0:5] + '_msk_' + files[i][10:13] + '.png'
        print("border masks name: " + border_masks_name)
        img = cv2.imread(imgs_dir + files[i])#.convert('L')
        im_c =img.copy()
        im = asarray(im_c)
        gray_scale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray_scale,20, 255, cv2.THRESH_BINARY) ###thresholding to distinquish interested region from the background
        thresh = thresh/255.
            # plt.imshow(thresh)
            # plt.show()

        thresh1 = thresh.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        morphed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

            ## Find the max-area contour
        cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea)[-1]

            ## Crop and save it
        x, y, w, h = cv2.boundingRect(cnt)
        dst = im_c[y:y + h, x:x + w]

        basename = os.path.basename(files[i])  # e.g. MyPhoto.jpg
        name = os.path.splitext(basename)[0]
        print('saving cropped img: ' + files[i])
        cv2.imwrite('./all_data/laikinai/cropped_img/' + name + '_sqr.tif', dst)
    # corresponding ground truth

        print("ground truth name: " + groundTruth_name)
    ##converting from gif to png:
        # g_truth = Image.open(groundTruth_dir + groundTruth_name)
        # converting to grayscale
        # g_truth.save(groundTruth_dir + groundTruth_name[0:13] +'.png')
    ############################
        g_truth = cv2.imread(groundTruth_dir +groundTruth_name)
        g = g_truth.copy()
    #groundTruth[i] = np.asarray(g)
        grt = g[y:y + h, x:x + w]
        basename = os.path.basename(groundTruth_name)  # e.g. MyPhoto.jpg
        name = os.path.splitext(basename)[0]
        print('saving cropped grt: ' +groundTruth_name)
        cv2.imwrite('./all_data/laikinai/cropped_grt/' + name + '_sqr.tif', grt)
    # corresponding border masks

        print("border masks name: " + border_masks_name)
        # b_mask = Image.open(borderMasks_dir + border_masks_name)
        # b_mask.save(borderMasks_dir + border_masks_name[0:13] + '.png')
    #####################################
        b_mask = cv2.imread(borderMasks_dir+ border_masks_name)
        b = b_mask.copy()
        #b_mask[i] = np.assaray(b)
        msk = b[y:y + h, x:x + w]
        basename = os.path.basename(border_masks_name)  # e.g. MyPhoto.jpg
        name = os.path.splitext(basename)[0]
        print('saving cropped msk: ' + border_masks_name)
        cv2.imwrite('./all_data/laikinai/cropped_msk/' + name + '_sqr.tif',msk)

crop_grt_msk(img_path, gth_path, msk_path)

# im = Image.open(gth_path + 'drive_grt_001.ext')
# im.show()
# im.save('drive_grt_001.png')

# im =cv2.imread('drive_grt_001.png')