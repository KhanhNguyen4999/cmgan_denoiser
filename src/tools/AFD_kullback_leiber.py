import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

def cosine_pairwise_similarities(A, B=None, eps=1e-6, normalized=True):
    """
    Calculates the pairwise cosine similarities between feature vectors
    :param A: Tensor of shape (batch_size, channels, height, width)
    :param B: Tensor of shape (batch_size, channels, height, width) or None
    :param eps: Small value to avoid division by zero
    :param normalized: Whether to normalize similarities to [0, 1]
    :return: Tensor of shape (batch_size, height, batch_size, height) with pairwise cosine similarities
    """
    if B is None:
        B = A
    
    # Normalize the feature vectors
    A_norm = torch.sqrt(torch.sum(A ** 2, dim=2, keepdim=True))
    A = A / (A_norm + eps)
    
    B_norm = torch.sqrt(torch.sum(B ** 2, dim=2, keepdim=True))
    B = B / (B_norm + eps)
    
    # Compute cosine similarity
    similarities = torch.bmm(A, B.transpose(1, 2))
    
    if normalized:
        similarities = (similarities + 1.0) / 2.0

    return similarities

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
        self.layer_weights = nn.Parameter(torch.rand(len(s_shapes)))  # Initializing layer weights
        self.prob_loss = ProbLoss()

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        layer_weights = F.softmax(self.layer_weights, dim=0)  # Apply softmax to layer weights
        student_feats_transform = self.linear_trans_s(student_feats)  # Transform student features

        for i, (student_f, teacher_f, weight) in enumerate(zip(student_feats_transform, teacher_feats, layer_weights)):
            loss += weight * self.prob_loss(student_f, teacher_f)  # Compute weighted loss

        return loss


class ProbLoss(nn.Module):
    def __init__(self):
        super(ProbLoss, self).__init__()
    
    def forward(self, student_f, teacher_f, epsilon=1e-6):
        bs, channel, t, f = teacher_f.size()
        # Permute to (batch_size, time, frequency, channels)
        student_f_time = student_f.permute(0, 3, 2, 1)
        teacher_f_time = teacher_f.permute(0, 3, 2, 1)
        # Reshape to (batch_size, time, channels * frequency)
        student_f_time = student_f_time.contiguous().view(bs * f, t, -1)
        teacher_f_time = teacher_f_time.contiguous().view(bs * f, t, -1)
        student_s1 = cosine_pairwise_similarities(student_f_time, normalized=True) # normalize to avoid the negative value, it would lead the nan value in log function
        teacher_s1 = cosine_pairwise_similarities(teacher_f_time, normalized=True)

         # Transform them into probabilities
        # student_s1 = student_s1 / torch.sum(student_s1, dim=1, keepdim=True)
        # teacher_s1 = teacher_s1 / torch.sum(teacher_s1, dim=1, keepdim=True)
        student_s1 = F.softmax(student_s1, dim=2)
        teacher_s1 = F.softmax(teacher_s1, dim=2)

        # print("--student: ", student_s1)
        # print("--teacher: ", teacher_s1)
        # Jeffrey's  combined
        
        loss = (teacher_s1 - student_s1) * (torch.log(teacher_s1) - torch.log(student_s1))

        print("---Loss: ", torch.mean(loss))
        return torch.mean(loss)

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


