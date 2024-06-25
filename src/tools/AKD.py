import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from models.conformer import Attention
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()
        
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        
        # Linear transformation and split into multiple heads
        Q = self.query(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to get dimensions (batch_size, num_heads, seq_len, head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        
        # Compute attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(energy, dim=-1)
        
        # Apply attention weights to value
        context = torch.matmul(attention_weights, V)
        
        # Reshape and transpose to get final context vector
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
        
        return context

class nn_bn_relu(nn.Module):
    def __init__(self, nin, nout):
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(False)

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.linear(x)

class Interpolate(nn.ModuleList):
    def __init__(self, out_shape, mode="nearest"):
        super(Interpolate, self).__init__()
        self.out_shape = out_shape
        self.mode = mode
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, self.out_shape, mode=self.mode)
        return x

def build_feature_connector_complex(s_channel, output_channel, out_shape):
    # mid_channel = np.min([128, s_channel // 2])
    C = [
        Interpolate(out_shape),
        nn.Conv2d(s_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False),
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

        # check lại chỗ này khi dùng toàn bộ layer của student
        self.feature_projection = nn.ModuleList([build_feature_connector_complex(s_shape[1], # student_channel
                                                                                t_shape[1], # teacher_channle
                                                                                (t_shape[2], t_shape[3])) # teacher H, teacher W
                                                                                for s_shape, t_shape in zip(s_shapes, t_shapes)])

    def forward(self, g_s):
        g_s = [operation(g_s[i]) for i, operation in enumerate(self.feature_projection)]
        return g_s


class MSELoss(nn.Module):
    def __init__(self, dim):
        super(MSELoss, self).__init__()
        self.attn_t = Attention(dim = dim,
                            heads = 2,
                            dim_head = 32,
                            dropout = 0.0).cuda()
        
        self.attn_f = Attention(dim = dim,
                            heads = 2,
                            dim_head = 32,
                            dropout = 0.0).cuda()
        
    def forward(self, student_feature, teacher_feature):
        b, c, t, f = student_feature.size()
        student_feature_t = student_feature.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        student_feature_t = self.attn_t(student_feature_t)
        student_feature_t = student_feature_t.view(b, f, t, c).permute(0, 3, 2, 1)
        loss_t = F.mse_loss(student_feature_t, teacher_feature)

        student_feature_f = student_feature.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)
        student_feature_f = self.attn_f(student_feature_f)
        student_feature_f = student_feature_f.view(b, t, f, c).permute(0, 3, 1, 2)
        loss_f = F.mse_loss(student_feature_f, teacher_feature)
        
        return loss_t + loss_f

class AKD(nn.Module):
    def __init__(self, s_shapes, t_shapes):
        super(AKD, self).__init__()
        self.linear_trans_s = LinearTransformStudent(s_shapes, t_shapes)  # Ensure this class is defined
        self.mse_loss = [MSELoss(t_shape[1]) for t_shape in t_shapes]
    
    def forward(self, student_feats, teacher_feats):
        student_feats_transform = self.linear_trans_s(student_feats)  # Transform student features
        loss = self.mse_loss[0](student_feats_transform[-1], teacher_feats[-1])
        # for i in range(len(self.mse_loss)):
        #     loss = self.mse_loss[i](student_feats_transform[-1], teacher_feats[-1])
        
        return loss
