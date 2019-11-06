from __future__ import absolute_import

from .mlp import *
from .multimodal import *
from .cnn import *


__model_factory = {
	'mlpsidenet': MLPSideNet,
	'mlprelationnet': MLPRelationSideNet,
	'ressidenet': ResSideNet,
	'mm_ge_cs': MMGECSNet,
	'mm_attention': MMAttentionNet,
	'mm_sum': MMSumNet,
	'mm_joint': MMJointNet,
	'mm_joint_attn': MMJointAttnNet,
	'mmm_sum': MMMSumNet,
	'mlpontology': MLPOntologyNet,
    'cnn': CNNNet
}


def get_names():
	return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
	if name not in list(__model_factory.keys()):
		raise KeyError("Unknown model: {}".format(name))
	return __model_factory[name](*args, **kwargs)