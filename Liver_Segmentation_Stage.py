# -*- coding: utf-8 -*-
"""
@author: Haneen Albarqi
"""
# Import the neccessary libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np

"""---------------------------stage4-------------------------------------
   ------------------Liver Segmentation Stage----------------------------
"""

class LiverSegmentationStage:
    def segmentLiver(self,O_img,img):
        
       # sure background area 
       kernel = np.ones((3,3),np.uint8)   # 3x3 kernel with all ones.
       sure_BG = cv2.dilate(img, kernel, iterations = 2)#3
       cv2.imshow('sure background ', sure_BG  ) 
       cv2.waitKey(0)
      
       # sure foreground area
       dist_Trans = cv2.distanceTransform(img,cv2.DIST_L2,5)
       ret, sure_FG = cv2.threshold(dist_Trans,0.7*dist_Trans.max(),255,0)
       cv2.imshow('sure foreground ',sure_FG) 
       cv2.waitKey(0)
       
       # Finding unknown region
       sure_FG = np.uint8(sure_FG)
       unknown = cv2.subtract(sure_BG,sure_FG)
       cv2.imshow('unknown ', unknown ) 
       cv2.waitKey(0)
       cv2.destroyAllWindows()

       # Marker labelling
       ret1, markers_lab = cv2.connectedComponents(sure_FG)

       # Add one to all labels so that sure background is not 0, but 1
       markers_lab = markers_lab+1

       # Now, mark the region of unknown with zero
       markers_lab[unknown==255] = 0
       markers_lab = cv2.watershed(O_img,markers_lab)
       O_img[markers_lab == -1] = [255,255,0] # can change the color of boundries [ , , ]
       plt.imshow(markers_lab)
       plt.show()	
       plt.imshow(O_img)
       plt.show()
       return O_img,markers_lab

    def extractROI(self, liver_img,markers) :
        
        m= cv2.convertScaleAbs(markers)
        #We threshold it properly to get the mask and perform bitwise_and with the input image:
        ret,pred_mask = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        res = cv2.bitwise_and(liver_img,liver_img,mask = pred_mask)
        return res,pred_mask