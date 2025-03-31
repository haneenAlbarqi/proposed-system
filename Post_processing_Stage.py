# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 07:05:10 2025

@author: Haneen Albarqi
"""

import matplotlib.pyplot as plt
import cv2
from Preprocessing_Stage import PreprocessingStage
from NS_domain_Stage import NS_domain_Stage
import skimage as ski
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
#from google.colab.patches import cv2_imshow # for image display
from PIL import Image,ImageFilter 
from numba import vectorize, guvectorize, float32, int32, jit, cuda 
from scipy import signal
from skimage import img_as_ubyte
import skimage.measure 
from scipy.stats import entropy 
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM 
import time
from skimage import morphology
from skimage.segmentation import clear_border

class postProcessingStage:
    
    def adaptive_thres(self, img):
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 199, 5) 
        adptvThresh_img = clear_border(thresh) #Remove edge touching grains
        return adptvThresh_img
        
    def morphologicalOP(self,img):
        kernel = np.ones((3,3),np.uint8)   # 3x3 kernel with all ones.
        cleaned = morphology.remove_small_objects(img, 200)
        removSmallObject = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel,iterations=2)
        cv2.imshow('removSmallObject',removSmallObject ) 	  
        cv2.waitKey(0)
         
        #_________________filling holes___________________
        im_fill = removSmallObject.copy()
         
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        L, W = removSmallObject .shape[:2]
        MASK = np.zeros((L+2, W+2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_fill, MASK , (0,0), 255); 
        # Invert floodfilled image
        im_fill_inv = cv2.bitwise_not(im_fill) 
        # Combine the two images to get the foreground.
        im_out = removSmallObject  | im_fill_inv
        cv2.imshow('fill holes ',im_out ) 	 
        cv2.waitKey(0)

        img_open1 = cv2.morphologyEx(im_out, cv2.MORPH_OPEN, kernel,iterations=2)
        cv2.imshow('opening ',img_open1 ) 	  
        cv2.waitKey(0)
        img_close1 = cv2.morphologyEx(img_open1, cv2.MORPH_CLOSE, kernel,iterations=1)  # Compare this image with the previous one
        cv2.imshow('opening ',img_close1 ) 	  
        cv2.waitKey(0)
    
        # Filter using contour area and remove small noise
        contourImg=img_close1
        cnts = cv2.findContours(contourImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 300:
                cv2.drawContours(contourImg, [c], -1, (0,0,0), -1)

        # Morph close and invert image
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        close = 255 - cv2.morphologyEx(contourImg, cv2.MORPH_CLOSE, kernel1, iterations=1)
      
        return contourImg


