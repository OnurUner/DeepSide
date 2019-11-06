import torch
import torch.nn as nn
from .subnet import SubNet, _make_block
import torch.nn.functional as F


class MLPSideNet(nn.Module):
    def __init__(self, in_dim, hid_layers, out_dim, p_dropout, **kwargs):
        super(MLPSideNet, self).__init__()
        self.network = SubNet(in_dim, hid_layers, p_dropout)
        self.clf = nn.Linear(hid_layers[-1], out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        output = self.network(X)
        output = self.clf(output)
        if not self.training:
            output = self.sigmoid(output)
        return output


class MLPRelationSideNet(nn.Module):
    def __init__(self, in_dim, hid_layers, out_dim, p_dropout, **kwargs):
        super(MLPRelationSideNet, self).__init__()
        self.network = SubNet(in_dim, hid_layers, p_dropout)
        self.clf1 = nn.Linear(hid_layers[-1], out_dim)
        self.clf2 = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        features = self.network(X)
        output1 = self.sigmoid(self.clf1(features))
        output2 = self.sigmoid(self.clf2(output1))
        if self.training:
            return output2, output1
        else:
            return output2


class MLPOntologyNet(nn.Module):
    def __init__(self, in_dim, hid_layers, out_dim, p_dropout, **kwargs):
        super(MLPOntologyNet, self).__init__()
        self.network = SubNet(in_dim, hid_layers, p_dropout)
        self.embedding_layer = nn.Linear(hid_layers[-1], 136)
        self.ontology_layer = nn.Linear(out_dim, out_dim)
        self.clf = nn.Sequential(SubNet(out_dim, [out_dim*5], p_dropout), nn.Linear(out_dim*5, out_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, ontology_vector):
        ontology_vector = F.tanh(self.ontology_layer(ontology_vector))
        output = self.network(X)
        output = self.embedding_layer(output)
        output = F.relu(torch.matmul(output, ontology_vector))
        output = self.clf(output)
        if not self.training:
            output = self.sigmoid(output)
        return output


class ResBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, p_dropout):
        super(ResBlock, self).__init__()
        up_blocks = _make_block(in_dim, hid_dim, p_dropout)
        dw_blocks = _make_block(hid_dim, in_dim, p_dropout)
        self.fc_up = nn.Sequential(*up_blocks)
        self.fc_dw = nn.Sequential(*dw_blocks)
        self.relu = nn.ReLU()

    def forward(self, X):
        res = X
        X = self.fc_up(X)
        X = self.fc_dw(X)

        X += res
        X = self.relu(X)
        return X


class ResSideNet(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim=2000, hid_dim=4000, n_res_block=3, p_dropout=0.5, **kwargs):
        super(ResSideNet, self).__init__()
        self.n_res_block = n_res_block
        self.hid_dim = hid_dim
        self.res_dim = res_dim
        self.p_dropout = p_dropout

        self.prep = nn.Linear(in_dim, res_dim)
        self.network = self._create_network()
        self.clf = nn.Linear(res_dim, out_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _create_network(self):
        network_blocks = []
        for i in range(self.n_res_block):
            network_blocks.append(ResBlock(self.res_dim, self.hid_dim, self.p_dropout))

        return nn.Sequential(*network_blocks)

    def forward(self, X):
        out = X
        out = self.prep(out)
        out = self.relu(out)
        out = self.network(out)
        out = self.clf(out)
        out = self.sigmoid(out)
        return out











