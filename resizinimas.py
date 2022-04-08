import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray

img_path = './all_data/laikinai/cropped_img/'
grt_path = './all_data/laikinai/cropped_grt/'
msk_path = './all_data/laikinai/cropped_msk/'

#--- DEFINE SIZE---:
new_size = (565, 565)

def resize_data(imgs_dir, groundTruth_dir, borderMasks_dir):
    files = os.listdir(imgs_dir)
    assert len(files) > 0
    img = cv2.imread(imgs_dir + files[0])
    for i in range(len(files)):
    # original
        print("original image: " + files[i])
    # corresponding ground truth
        groundTruth_name = files[i][0:5] + '_grt_' + files[i][10:13] + '_sqr.tif'
        print("ground truth name: " + groundTruth_name)
        border_masks_name = files[i][0:5] + '_msk_' + files[i][10:13] + '_sqr.tif'
        print("border masks name: " + border_masks_name)
        print('Resizing images')
    # Resizing images
        im = cv2.imread(imgs_dir + files[i])
        im_c = im.copy()
    # - NEAREST NEIGHBOUR -#
        new_im = cv2.resize(im_c,new_size, interpolation =cv2.INTER_NEAREST)
        basename = os.path.basename(files[i])  # e.g. MyPhoto.jpg
        name = os.path.splitext(basename)[0]
        print("saving " + files[i] + " near_neigh")
        cv2.imwrite('./all_data/train/images/sqr/565_565/neigh/'+ name + '_neigh.tif' , new_im)
    # - LINEAR -#
        new_im = cv2.resize(im_c, new_size, interpolation=cv2.INTER_LINEAR)
        print("saving " + files[i] + " linear")
        cv2.imwrite('./all_data/train/images/sqr/565_565/linear/' + name + '_liner.tif', new_im)
    # - BICUBIC -#
        new_im = cv2.resize(im_c, new_size, interpolation=cv2.INTER_CUBIC)
        print("saving " + files[i] + " bicubic")
        cv2.imwrite('./all_data/train/images/sqr/565_565/cubic/' + name + '_cubic.tif', new_im)

        ### all same to grt
        print('Resizing ground truths')
        g_truth = cv2.imread(groundTruth_dir + groundTruth_name)
        g = g_truth.copy()
        new_im = cv2.resize(g, new_size, interpolation=cv2.INTER_NEAREST)
        basename = os.path.basename(groundTruth_name)  # e.g. MyPhoto.jpg
        name = os.path.splitext(basename)[0]
        print("saving " + groundTruth_name + " near_neigh")
        cv2.imwrite('./all_data/train/manual/sqr/565_565/neigh/' + name + '_neigh.tif', new_im)
        new_im = cv2.resize(g, new_size, interpolation=cv2.INTER_LINEAR)
        print("saving " + groundTruth_name + " linear")
        cv2.imwrite('./all_data/train/manual/sqr/565_565/linear/' + name + '_liner.tif', new_im)
        new_im = cv2.resize(g, new_size, interpolation=cv2.INTER_CUBIC)
        print("saving " + groundTruth_name + " bicubic")
        cv2.imwrite('./all_data/train/manual/sqr/565_565/cubic/' + name + '_cubic.tif', new_im)

        ### all same to msks
        print('Resizing border masks')
        b_mask = cv2.imread(borderMasks_dir + border_masks_name)
        b = b_mask.copy()
        new_im = cv2.resize(b, new_size, interpolation=cv2.INTER_NEAREST)
        basename = os.path.basename(border_masks_name)  # e.g. MyPhoto.jpg
        name = os.path.splitext(basename)[0]
        print("saving " + border_masks_name + " near_neigh")
        cv2.imwrite('./all_data/train/masks/sqr/565_565/neigh/' + name + '_neigh.tif', new_im)
        new_im = cv2.resize(b, new_size, interpolation=cv2.INTER_LINEAR)
        print("saving " + border_masks_name + " linear")
        cv2.imwrite('./all_data/train/masks/sqr/565_565/linear/' + name + '_liner.tif', new_im)
        new_im = cv2.resize(b, new_size, interpolation=cv2.INTER_CUBIC)
        print("saving " + border_masks_name + " bicubic")
        cv2.imwrite('./all_data/train/masks/sqr/565_565/cubic/' + name + '_cubic.tif', new_im)

resize_data(img_path, grt_path, msk_path)