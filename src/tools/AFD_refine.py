import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

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
    def __init__(self, s_shapes, t_shapes):
        super(AFD, self).__init__()
        self.linear_trans_s = LinearTransformStudent(s_shapes, t_shapes)  # Ensure this class is defined
        self.channel_diff = nn.ModuleList([DiffByChannel(t_layer_shape[1]) for t_layer_shape in t_shapes])  # Ensure this class is defined
        self.layer_weights = nn.Parameter(torch.rand(len(s_shapes)))  # Initializing layer weights

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        layer_weights = F.softmax(self.layer_weights, dim=0)  # Apply softmax to layer weights
        
        student_feats_transform = self.linear_trans_s(student_feats)  # Transform student features

        for i, (student_f, teacher_f, weight) in enumerate(zip(student_feats_transform, teacher_feats, layer_weights)):
            loss += weight * self.channel_diff[i](student_f, teacher_f)  # Compute weighted loss

        # print(loss)
        return loss

class DiffByChannel(nn.Module):
    def __init__(self, d):
        super(DiffByChannel, self).__init__()
        self.weight = nn.Parameter(torch.randn(d))
        self.weight.requires_grad = True
    
    def forward(self, student_f, teacher_f):
        bs, channel, t, f = teacher_f.size()
        teacher_f = teacher_f.view(bs, channel, -1)
        student_f = student_f.view(bs, channel, -1)

        diff = (teacher_f - student_f).pow(2).mean(2)
        diff = torch.mul(diff, F.softmax(self.weight, dim=0)).sum(1)
        return torch.sum(diff)

class Interpolate(nn.ModuleList):
    def __init__(self, out_shape, mode="nearest"):
        super(Interpolate, self).__init__()
        self.out_shape = out_shape
        self.mode = mode
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, self.out_shape, mode=self.mode)
        return x
    
class SelfAttention(nn.ModuleList):
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
        # nn.BatchNorm2d(output_channel)
        ]
    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

class LinearTransformStudent(nn.Module):
    def __init__(self, s_shapes, t_shapes):
        super(LinearTransformStudent, self).__init__()
        self.feature_projection = nn.ModuleList([build_feature_connector_complex(s_shape[1], # student_channel
                                                                                t_shape[1], # teacher_channle
                                                                                (t_shape[2], t_shape[3])) # teacher H, teacher W
                                                                                for s_shape, t_shape in zip(s_shapes, t_shapes)])

    def forward(self, g_s):
        g_s = [operation(g_s[i]) for i, operation in enumerate(self.feature_projection)]
        return g_s


