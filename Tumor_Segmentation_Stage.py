# -*- coding: utf-8 -*-
"""
@author: Haneen Albarqi
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture as GMM 


class TumorSegmentationStage:

    def segmentTumor(self,res):
        # Convert MxNx3 image into Kx3 where K=MxN
        img2 = res.reshape((-1,3))  #-1 reshape means, in this case MxN

        #for GMM cluster
        #covariance choices, full, tied, diag, spherical
        gmm_model = GMM(n_components=3, covariance_type='full').fit(img2)  #tied works better than full
        gmm_labels = gmm_model.predict(img2)
        #Put numbers back to original shape so we can reconstruct segmented image
        original_shape = res.shape
        segmented_T = gmm_labels.reshape(original_shape[0], original_shape[1])
        segmented_T = np.expand_dims(segmented_T, axis=-1)
        return segmented_T
       
    def extractTumor(self,img,res):
        #for foreground and background segmentation using GMM
        foreground1 = np.uint8(np.multiply(img, res))
        background1 = np.uint8(res - foreground1)
        plt.imshow(foreground1) 
        plt.show()
        plt.imshow(background1)
        plt.show()
        
        kernel = np.ones((3,3),np.uint8)   # 3x3 kernel with all ones.
        closeBack = cv2.morphologyEx(foreground1, cv2.MORPH_CLOSE, kernel,iterations=1)
        openBack = cv2.morphologyEx(closeBack, cv2.MORPH_OPEN, kernel,iterations=1)

        finalToumer = np.uint8(np.bitwise_and(openBack, background1))
        return finalToumer
