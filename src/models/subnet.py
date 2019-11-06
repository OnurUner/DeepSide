import torch
import torch.nn as nn


def _make_block(D_in, D_out, p_dropout=0.5):
	linear = nn.Linear(D_in, D_out)
	torch.nn.init.xavier_normal_(linear.weight)
	norm = nn.BatchNorm1d(D_out)
	activation = nn.ReLU(inplace=True)
	dropout = nn.Dropout(p=p_dropout) # 0.4 for mm models # 0.3 perfect for mlp
	layers = [linear, norm, activation, dropout]
	return layers
	

class SubNet(nn.Module):
	def __init__(self, dim_in, hid_layer_dims, p_dropout):
		super(SubNet, self).__init__()
		self.dim_in = dim_in
		self.dim_out = hid_layer_dims[-1]
		self.hid_layer_dims = hid_layer_dims
		self.p_dropout = p_dropout
		
		network_blocks = self.create_network()
		self.network = nn.Sequential(*network_blocks)

	def create_network(self):
		network_blocks = []
		network_blocks.extend(_make_block(self.dim_in, self.hid_layer_dims[0], self.p_dropout))
		for i in range(1, len(self.hid_layer_dims)):
			network_blocks.extend(_make_block(self.hid_layer_dims[i-1], self.hid_layer_dims[i], self.p_dropout))
		return network_blocks
	
	def forward(self, X):
		X = self.network(X)
		return X
	