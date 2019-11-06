import torch
import torch.nn as nn
import torch.nn.functional as F
from .subnet import SubNet


class MMGECSNet(nn.Module):
    def __init__(self, ge_in, ge_layers, cs_in, cs_layers, out_dim, p_dropout, **kwargs):
        super(MMGECSNet, self).__init__()
        self.ge_net = SubNet(ge_in, ge_layers, p_dropout)
        self.cs_net = SubNet(cs_in, cs_layers, p_dropout)
        self.feature_size = self.ge_net.dim_out + self.cs_net.dim_out
        self.clf = nn.Linear(self.feature_size, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_ge, X_cs):
        out_ge = self.ge_net(X_ge)
        out_cs = self.cs_net(X_cs)
        output = torch.cat((out_ge, out_cs), 1)
        output = self.clf(output)
        output = self.sigmoid(output)
        return output


class MMAttentionNet(nn.Module):
    def __init__(self, ge_in, cs_in, ge_layers, cs_layers, feature_dim, cls_layers, out_dim, p_dropout, **kwargs):
        super(MMAttentionNet, self).__init__()
        self.feature_dim = feature_dim

        self.ge_net = SubNet(ge_in, ge_layers, p_dropout)
        self.ge_emb = nn.Linear(self.ge_net.dim_out, self.feature_dim)
        self.ge_bn  = nn.BatchNorm1d(self.feature_dim)

        self.cs_net = SubNet(cs_in, cs_layers, p_dropout)
        self.cs_emb = nn.Linear(self.cs_net.dim_out, self.feature_dim)
        self.cs_bn  = nn.BatchNorm1d(self.feature_dim)
        #self.att = Attention(self.feature_dim)

        self.norm = nn.BatchNorm1d(self.feature_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.ge_clf = nn.Linear(self.feature_dim, out_dim)
        #self.cs_clf = nn.Linear(self.feature_dim, out_dim)
        #self.clf = nn.Sequential(SubNet(self.feature_dim, cls_layers),
        #                         nn.Linear(cls_layers[-1], out_dim))
        self.clf = nn.Linear(self.feature_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_ge, X_cs):
        out_ge = self.ge_net(X_ge)
        out_ge = self.ge_emb(out_ge)
        out_ge = self.ge_bn(out_ge)
        #out_ge = self.relu(out_ge)

        out_cs = self.cs_net(X_cs)
        out_cs = self.cs_emb(out_cs)
        out_cs = self.cs_bn(out_cs)
        #out_cs = self.tanh(out_cs)

        output = out_ge + out_cs
        output = self.relu(output)
        #output = self.att(out_cs, out_ge)

        output = self.clf(output)
        #return self.clf(output)
        if self.training:
            return output
        else:
            return self.sigmoid(output)


class MMSumNet(nn.Module):
    def __init__(self, ge_in, cs_in, ge_layers, cs_layers, out_dim, p_dropout, **kwargs):
        super(MMSumNet, self).__init__()
        self.ge_net = SubNet(ge_in, ge_layers, p_dropout)
        self.cs_net = SubNet(cs_in, cs_layers, p_dropout)

        self.clf = nn.Sequential(nn.Linear(ge_layers[-1], out_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_ge, X_cs):
        out_ge = self.ge_net(X_ge)

        out_cs = self.cs_net(X_cs)

        output = out_ge + out_cs

        output = self.clf(output)
        if not self.training:
            output = self.sigmoid(output)
        return output


class MMMSumNet(nn.Module):
    def __init__(self, ge_in, cs_in, meta_in, ge_layers, cs_layers, meta_layers, out_dim, p_dropout, **kwargs):
        super(MMMSumNet, self).__init__()
        self.ge_net = SubNet(ge_in, ge_layers, p_dropout)
        self.cs_net = SubNet(cs_in, cs_layers, p_dropout)
        self.meta_net = SubNet(meta_in, meta_layers, p_dropout)

        #self.clf = nn.Sequential(nn.Linear(3*ge_layers[-1], out_dim))
        self.clf = nn.Sequential(nn.Linear(ge_layers[-1], out_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_ge, X_cs, X_meta):
        out_ge = self.ge_net(X_ge)
        out_cs = self.cs_net(X_cs)
        out_meta = self.meta_net(X_meta)
        output = out_ge + out_cs + out_meta
        #output = torch.cat((out_cs, out_ge), 1)
        #output = torch.cat((output, out_meta), 1)

        output = self.clf(output)
        if not self.training:
            output = self.sigmoid(output)
        return output


class GE_Attention(nn.Module):
    """ Encoder attention module"""
    def __init__(self, ge_in, cs_in):
        super(GE_Attention, self).__init__()
        self.cs_w = nn.Linear(cs_in, cs_in)
        self.ge_w = nn.Linear(ge_in, cs_in)
        self.psi = nn.Sequential(
            nn.Linear(cs_in, cs_in),
            nn.BatchNorm1d(cs_in),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(cs_in)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cs_x, ge_x):
        cs = self.cs_w(cs_x)
        ge = self.ge_w(ge_x)
        psi = self.bn(cs * ge)
        # psi = self.relu(ge_x + cs_x)
        psi = self.psi(psi)
        return psi


class MMJointAttnNet(nn.Module):
    def __init__(self, ge_in, cs_in, ge_layers, cs_layers, out_dim, p_dropout, **kwargs):
        super(MMJointAttnNet, self).__init__()

        self.ge_net = SubNet(ge_in, ge_layers, p_dropout)
        self.cs_net = SubNet(cs_in, cs_layers, p_dropout)

        self.ge_clf = nn.Linear(self.ge_net.dim_out, out_dim)
        self.cs_clf = nn.Linear(self.cs_net.dim_out, out_dim)

        self.mm_net = SubNet(self.cs_net.dim_out, cs_layers, p_dropout)
        self.mm_clf = nn.Linear(self.mm_net.dim_out, out_dim)

        self.attn = GE_Attention(self.ge_net.dim_out, self.cs_net.dim_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_ge, X_cs):
        out_ge = self.ge_net(X_ge)
        out_cs = self.cs_net(X_cs)
        # out_mm = self.attn(out_cs, out_ge)
        out_mm = self.sigmoid(out_ge + out_cs)

        out_mm = self.mm_clf(out_mm)
        out_ge = self.ge_clf(out_ge)
        out_cs = self.cs_clf(out_cs)
        if self.training:
            return out_mm, out_ge, out_cs
            # return out_mm
        else:
            return self.sigmoid(out_mm)


class MMJointNet(nn.Module):
    def __init__(self, ge_in, cs_in, ge_layers, cs_layers, out_dim, p_dropout, **kwargs):
        super(MMJointNet, self).__init__()

        self.ge_net = SubNet(ge_in, ge_layers, p_dropout)
        self.cs_net = SubNet(cs_in, cs_layers, p_dropout)

        self.ge_clf = nn.Linear(self.ge_net.dim_out, out_dim)
        self.cs_clf = nn.Linear(self.cs_net.dim_out, out_dim)
        self.mm_clf = nn.Linear(out_dim, out_dim)

        self.attention = GE_Attention(out_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_ge, X_cs):
        out_ge = self.ge_net(X_ge)
        out_cs = self.cs_net(X_cs)

        out_ge = self.sigmoid(self.ge_clf(out_ge))
        out_cs = self.sigmoid(self.cs_clf(out_cs))
        out_mm = self.attention(out_ge, out_cs)
        # out_mm = self.bn(out_mm)
        # out_mm = self.sigmoid(self.mm_clf(out_mm))
        if self.training:
            return out_mm, out_ge, out_cs
        else:
            return out_mm