# -*- coding: utf-8 -*-
"""
@author: Haneen Albarqi
"""
import numpy as np
import pandas as pd
from IPython.display import display

class EvaluateSegmentation:

	def __init__(self, GT_mask, predic_mask):
		self.GT_mask = GT_mask.astype(bool)
		self.predic_mask = predic_mask.astype(bool)
		self.intersection = self.GT_mask * self.predic_mask
		self.union = self.GT_mask + self.predic_mask
		self.true_positive = self.intersection
		self.false_positive = self.union != self.GT_mask
		self.false_negative = self.union != self.predic_mask
		self.true_negative = np.invert(self.union)
        

	def Acc(self):
		return np.sum(self.true_positive+self.true_negative)/np.sum(self.true_positive+self.true_negative+self.false_negative+self.false_positive)

	def Precision(self):
		return np.sum(self.true_positive)/np.sum(self.true_positive+self.false_positive)

	def Recall(self):
		return np.sum(self.true_positive)/np.sum(self.true_positive+self.false_negative)

	def Dice(self):
		return 2*np.sum(self.true_positive)/(2*np.sum(self.true_positive)+np.sum(self.false_positive)+np.sum(self.false_negative))

	def Jaccard(self):
		return np.sum(self.intersection)/np.sum(self.union)

	def confusion_matrix(self):
		GT_series = pd.Series(self.GT_mask.flatten(), name="Ground truth")
		predic_series = pd.Series(self.predic_mask.flatten(), name="Predicted")
		df_confusion = pd.crosstab(GT_series, predic_series)
		display(df_confusion)


