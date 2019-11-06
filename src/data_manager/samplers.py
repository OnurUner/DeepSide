from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import numpy as np
import copy
import random

import torch
from torch.utils.data.sampler import Sampler


class RandomPertSampler(Sampler):

	def __init__(self, data_source):
		self.data_source = data_source
		self.drug_ids = data_source.drug_ids
		self.index_dic = defaultdict(list)
		for index, d_id in enumerate(self.drug_ids):
			self.index_dic[d_id].append(index)

		self.length = len(self.index_dic.keys())

	def __iter__(self):
		epoch_idxs = []
		for d_id in self.index_dic:
			epoch_idxs.extend(random.sample(self.index_dic[d_id], 1))
		random.shuffle(epoch_idxs)
		return iter(epoch_idxs)

	def __len__(self):
		return self.length