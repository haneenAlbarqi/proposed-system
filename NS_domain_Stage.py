# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 05:54:10 2025

@author: Haneen Albarqi
"""

# Import the neccessary libraries
import cv2
import numpy as np
#from google.colab.patches import cv2_imshow # for image display
from numba import vectorize, jit 
from scipy import signal

class NS_domain_Stage:
    

   def Neutrsophic_Image(self,image):
       
       def img_filter(img):
           window_size = 3
           window_Val = 1/window_size
           conCol_Matrix = np.zeros(shape=(window_size,1))
           conCol_Matrix = np.add(conCol_Matrix, window_Val)
           conRow_Matrix = np.zeros(shape=(1,window_size))
           conRow_Matrix  = np.add(conRow_Matrix, window_Val)
           imgfilter = signal.convolve2d(img, conCol_Matrix, boundary='symm', mode='same')
           return signal.convolve2d(imgfilter, conRow_Matrix, boundary='symm', mode='same')

       @jit(nopython=True)
       def Neutrsophic_set(subset, min_Val, max_Val):
           return (subset - min_Val) / (max_Val - min_Val)   
       
       # get True subset
       lmg_v = img_filter(image)
       lmg_v_Min = np.amin(lmg_v)
       lmg_v_Max = np.amax(lmg_v)
       NS_TIsubset= np.vectorize(Neutrsophic_set)
       T = NS_TIsubset(lmg_v,lmg_v_Min,lmg_v_Max)
       # get indeterminacy subset
       absMatrix = np.absolute(np.array(image) - np.array(lmg_v))
       absMatrix_Min = np.amin(absMatrix)
       absMatrix_Max = np.amax(absMatrix)
       I = NS_TIsubset(absMatrix, absMatrix_Min, absMatrix_Max)
       F = 1 - T    
       TFI = np.zeros((len(T),len(T[0]),3))
       TFI[:,:,0] = T
       TFI[:,:,1] = I
       TFI[:,:,2] = F
       
       cv2.imshow("TRUE",T)
       cv2.waitKey(0)
          
       cv2.imshow("I",I) 
       cv2.waitKey(0)
       
       cv2.imshow("FALS",F)
       cv2.waitKey(0)
       
       return TFI
        
  

