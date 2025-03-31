# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 03:02:48 2025

@author: Haneen Albarqi
"""

import cv2

class PreprocessingStage:
    def preProcessing(self,img):
                  
        # convert to gray scale
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # median filter with 3×3 window
        window_size=3
        median_img = cv2.medianBlur(img_gray, window_size)
        
        # Histogram equalization
        Histogram_img=cv2.equalizeHist(median_img)
        return Histogram_img
    
    
        
        
        
        