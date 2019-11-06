from __future__ import absolute_import

from .data_mlp import *
from .data_multimodal import *
from .data_cnn import *


__model_factory = {
	'CS': get_cs,
	'GE': get_ge,
	'GE_CS_META': get_ge_cs_meta,
	'Multiple': get_dataset,
	'GO': get_processed_data,
	'CD': get_processed_data,
    'IMAGE': get_cs_images
}


def get_names():
	return list(__model_factory.keys())


def init_dataset(name, *args, **kwargs):
	if name not in list(__model_factory.keys()):
		raise KeyError("Unknown model: {}".format(name))
	return __model_factory[name](*args, **kwargs)