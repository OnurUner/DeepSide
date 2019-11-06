# -*- coding: utf-8 -*-
import numpy as np
from src.utils.metrics import meter
from src.utils.metrics.aucmeter import AUCMeter


class RocMeter(meter.Meter):

	def __init__(self):
		super(RocMeter, self).__init__()
		self.reset()

	def reset(self):
		self.auc_meters = []

	def add(self, output, target):
		for i in range(target.shape[1]):
			self.auc_meters.append(AUCMeter())
			self.auc_meters[i].add(output[:,i], target[:, i])

	def value(self):
		auc_scores = []
		for auc_meter in self.auc_meters:
			area, _, _, = auc_meter.value()
			auc_scores.append(area)
		return np.mean(auc_scores)