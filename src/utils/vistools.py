# -*- coding: utf-8 -*-
from __future__ import absolute_import

import random
import pandas as pd

def bar_chart_csv(path, num_of_drug):
	df = pd.read_csv(path, index_col=0)
	drug_ids = df.loc[:,'drug_id'].values.tolist()
	drug_ids = random.sample(drug_ids, num_of_drug)
	df = df.loc[df['drug_id'].isin(drug_ids)]
	return df, drug_ids