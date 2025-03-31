# -*- coding: utf-8 -*-
"""
@author: Haneen Albarqi
"""

import matplotlib.pyplot as plt
import cv2
from Preprocessing_Stage import PreprocessingStage
from NS_domain_Stage import NS_domain_Stage
from Post_processing_Stage import postProcessingStage
from Liver_Segmentation_Stage import LiverSegmentationStage
from Tumor_Segmentation_Stage import TumorSegmentationStage
from evaluate import EvaluateSegmentation
import skimage as ski
from skimage import img_as_ubyte
import skimage.measure 



original_img=cv2.imread('train/Images/image_89.png')
groundTruth_mask= cv2.imread('train\Masks\mask_89.png')
cv2.imshow("original image", original_img)
cv2.waitKey(0) 
cv2.imshow("ground truth mask", groundTruth_mask)
cv2.waitKey(0) 

resized_img=cv2.resize(original_img,(128,128))
groundTruth_mask=cv2.resize(groundTruth_mask,(128,128))
groundTruth_mask=cv2.cvtColor(groundTruth_mask,cv2.COLOR_BGR2GRAY)

""" ------------------------- Preprocessing Stage -----------------------------
"""
preprocessing_img=PreprocessingStage() #Create an object from the class
enhanced_img = preprocessing_img.preProcessing(resized_img)

cv2.imshow("enhanced_img", enhanced_img)
cv2.waitKey(0) 

""" ----------------- Conversion into NS domain Stage -------------------------
"""
NS_domain= NS_domain_Stage() #Create an object from the class
Img_NS = NS_domain.Neutrsophic_Image(enhanced_img)

#convert T,F&I to uint8 to can save them as jpg image
binaryT = Img_NS[:,:,0] 
T_img_8bit = img_as_ubyte(binaryT)

binaryI = Img_NS[:,:,1]
I_img_8bit = img_as_ubyte(binaryI)

binaryF = Img_NS[:,:,2]
F_img_8bit = img_as_ubyte(binaryF)

threshold_value = ski.filters.threshold_otsu(T_img_8bit)
print(f"threshold_value: {threshold_value}")
T_binaryimg = cv2.threshold(T_img_8bit, threshold_value, 255, cv2.THRESH_BINARY)[1]


""" --------------------------Post-processing Stage----------------------------
"""
postProcessing= postProcessingStage() #Create an object from the class

# adaptive thresholding
adptvThresh_img=postProcessing.adaptive_thres(T_binaryimg)
cv2.imshow('Adaptive Gaussian', adptvThresh_img) 
cv2.waitKey(0)

# morphological
morph_img=postProcessing.morphologicalOP(adptvThresh_img)
cv2.imshow('morphological Img', morph_img)
cv2.waitKey(0)

""" ----------------------Liver Segmentation Stage-----------------------------
"""
liverSegmentaion=LiverSegmentationStage()  #Create an object from the class
SegLiver_img, marker=liverSegmentaion.segmentLiver(resized_img,morph_img)
ROI_img,pred_mask=liverSegmentaion.extractROI(SegLiver_img, marker)
plt.imshow(ROI_img)
plt.show()
plt.imshow(pred_mask)
plt.show()

""" ----------------------Tumor Segmentation Stage-----------------------------
"""
TumorSegmentaion=TumorSegmentationStage()  #Create an object from the class
Tumor_img=TumorSegmentaion.segmentTumor(ROI_img)
seg_Tumor_img=TumorSegmentaion.extractTumor(Tumor_img,ROI_img)
plt.imshow(seg_Tumor_img)
plt.show()
plt.imshow(Tumor_img)
plt.show()

""" -----------------------------evaluate--------------------------------------
"""
evaluateMetrix = EvaluateSegmentation(groundTruth_mask, pred_mask)

print(f"Accuracy: {round(evaluateMetrix.Acc(),4)}")
print(f"Precision: {round(evaluateMetrix.Precision(),4)}")
print(f"Recall: {round(evaluateMetrix.Recall(),4)}")
print(f"Dice: {round(evaluateMetrix.Dice(), 4)}")
print(f"Jaccard: {round(evaluateMetrix.Jaccard(), 4)}")
print("Confusion Matrix:")
evaluateMetrix.confusion_matrix()
