import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray

mask_train = "./DRIVE/training/mask/"

for image in os.listdir(mask_train):
    if image.endswith('.gif'):
        im = Image.open(mask_train + image).convert('RGB')
        im_c = im.copy()
        im_c = asarray(im_c)
        gray_scale = cv2.cvtColor(im_c, cv2.COLOR_BGR2GRAY)
        #
        _, thresh = cv2.threshold(gray_scale,20, 255, cv2.THRESH_BINARY)
        thresh = thresh/255.
        # plt.imshow(thresh)
        # plt.show()
        #
        thresh1 = thresh.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        morphed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

        ##LEAVING ONLY FOV
        ## the max-area contour
        cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea)[-1]

        ## Crop and save it
        x, y, w, h = cv2.boundingRect(cnt)
        dst = im_c[y:y + h, x:x + w]
        # plt.imshow(dst)
        # plt.show()
        #
        basename = os.path.basename(image)  # e.g. MyPhoto.jpg
        name = os.path.splitext(basename)[0]  # e.g. MyPhoto
        cv2.imwrite('./DRIVE/training/mask/cropped/' + name[0:2] + '_drive_msk_sqr.tif', dst)


        new_size = (2260,2260)
        print(im_c.shape)
        print(dst.shape)
        new_im = cv2.resize(dst,new_size, interpolation =cv2.INTER_CUBIC)
        #new_name = os.path.join('./DRHAGIS/test/images/cropped/565x584/', "sized_" + image)
        cv2.imwrite('./DRIVE/training/mask/cropped/2260x2260/' + name[0:2] + '_drive_msk_sqr2_r_cubic.tif', new_im)
