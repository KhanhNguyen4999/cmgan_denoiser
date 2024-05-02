# Attention-based Feature-level Distillation 
# Copyright (c) 2021-present NAVER Corp.
# Apache License v2.0

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import math


class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))


class AFD(nn.Module):
    def __init__(self, t_shapes, s_shapes, qk_dim):
        super(AFD, self).__init__()

        # Define fix local variable
        self.qk_dim = qk_dim
        self.n_t = len(t_shapes)
        self.linear_trans_s = LinearTransformStudent(t_shapes, s_shapes, qk_dim)
        self.linear_trans_t = LinearTransformTeacher(t_shapes, qk_dim)

        self.p_t = nn.Parameter(torch.Tensor(len(t_shapes), qk_dim))
        self.p_s = nn.Parameter(torch.Tensor(len(s_shapes), qk_dim))
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
    
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)
        query, h_t_all = self.linear_trans_t(g_t)

        p_logit = torch.matmul(self.p_t, self.p_s.t())

        logit = torch.add(torch.einsum('bstq,btq->bts', bilinear_key, query), p_logit) / np.sqrt(self.qk_dim)
        atts = F.softmax(logit, dim=2)  # b x t x s
        loss = 0.0

        for i in range(self.n_t):
            h_hat_s = h_hat_s_all[i]
            h_t = h_t_all[i]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss += diff

        return loss, atts

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff


class LinearTransformTeacher(nn.Module):
    def __init__(self, t_shapes, qk_dim):
        super(LinearTransformTeacher, self).__init__()
        self.query_layer = nn.ModuleList([nn_bn_relu(t_shape[1], qk_dim) for t_shape in t_shapes])
        self.channel_wise_operation = nn.ModuleList([nn.Conv2d(t_shape[1], 1, (3,3), padding=1) for t_shape in t_shapes])

    def forward(self, g_t):
        bs = g_t[0].size(0)
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = [operation(g_t[i]).view(bs, -1) for i, operation in enumerate(self.channel_wise_operation)] 
        # spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]

        query = torch.stack([query_layer(f_t, relu=False) for f_t, query_layer in zip(channel_mean, self.query_layer)],
                            dim=1)
        # value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]
        return query, spatial_mean


class LinearTransformStudent(nn.Module):
    def __init__(self, t_shapes, s_shapes, qk_dim):
        super(LinearTransformStudent, self).__init__()
        self.t = len(t_shapes)
        self.s = len(s_shapes)
        self.qk_dim = qk_dim
        self.relu = nn.ReLU(inplace=False)
        self.samplers = nn.ModuleList([Sample(t_shape, s_shapes) for t_shape in t_shapes])

        self.key_layer = nn.ModuleList([nn_bn_relu(s_shape[1], qk_dim) for s_shape in s_shapes])
        self.bilinear = nn_bn_relu(qk_dim, qk_dim * len(t_shapes))

    def forward(self, g_s):
        bs = g_s[0].size(0)
        channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        key = torch.stack([key_layer(f_s) for key_layer, f_s in zip(self.key_layer, channel_mean)],
                                     dim=1).view(bs * self.s, -1)  # Bs x h
        bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)
        # value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]
        return bilinear_key, spatial_mean


class Interpolate(nn.Module):
    def __init__(self, out_shape, mode="nearest"):
        super(Interpolate, self).__init__()
        self.out_shape = out_shape
        self.mode = mode
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, self.out_shape, mode=self.mode)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_channel):
        super(SelfAttention, self).__init__()
        self.conv = nn.Conv2d(input_channel, 1, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        att = self.conv(x)
        x = x * att
        return x

def build_feature_connector_complex(s_channel, output_channel, out_shape):
    mid_channel = np.min([128, s_channel // 2])
    C = [nn.Conv2d(s_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(mid_channel),
        nn.ReLU(),
        SelfAttention(mid_channel),
        nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(mid_channel),
        nn.ReLU(),
        Interpolate(out_shape),
        nn.Conv2d(mid_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(output_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

class Sample(nn.Module):
    def __init__(self, t_shape, s_shapes):
        super(Sample, self).__init__()
        t_N, t_C, t_H, t_W = t_shape
        self.feature_projection = nn.ModuleList([build_feature_connector_complex(s_shape[1], 1, (t_H, t_W)) for s_shape in s_shapes])

    def forward(self, g_s, bs):
        g_s = torch.stack([operation(g_s[i]).view(bs, -1) for i, operation in enumerate(self.feature_projection)], dim=1)
        return g_s