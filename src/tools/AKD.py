import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from models.conformer import Attention

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

class AKD(nn.Module):
    def __init__(self, args):
        super(AKD, self).__init__()
        # (b, 256, 40, 25) -> (b*t, 25, 256)
        # (b, 64, 321, 101)
       # hardcode get the first layer in student and teacher
        self.attn_t = Attention(dim = args.s_shapes[0][1],
                            heads = 2,
                            dim_head = 128,
                            dropout = 0.0)
        
        self.attn_f = Attention(dim = args.s_shapes[0][1],
                            heads = 2,
                            dim_head = 128,
                            dropout = 0.0)
        
        self.up = torch.nn.Upsample(size=(args.t_shapes[0][2], args.t_shapes[0][3]), 
                                        mode='bilinear', 
                                        align_corners=False)
        
        self.ff_ts = nn_bn_relu(args.s_shapes[0][1], 256)
        self.ff_tt = nn_bn_relu(args.t_shapes[0][1], 256)

        self.ff_fs = nn_bn_relu(args.s_shapes[0][1], 256)
        self.ff_ft = nn_bn_relu(args.t_shapes[0][1], 256)

    def kullback_leibler_loss(self, student_outputs, teacher_outputs, T=1.0, alpha=0.5):
        """
        Calculate the knowledge distillation loss using KL divergence.
        :param student_outputs: Outputs of the student model
        :param teacher_outputs: Outputs of the teacher model
        :param T: Temperature parameter (default: 1.0)
        :param alpha: Weight of the KL divergence term (default: 0.5)
        """
        
        kd_loss = nn.KLDivLoss()(nn.functional.log_softmax(student_outputs / T, dim=1),
                                nn.functional.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T)
        return kd_loss
    
    def forward(self, g_s, g_t):
        # hardcode get the first layer in student and teacher
        student_feature = g_s[0]
        student_feature = self.up(student_feature)
        b, c, t, f = student_feature.size()
        student_feature = student_feature.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        student_feature = self.attn_t(student_feature)
    
        student_feature_ff = self.ff_ts(student_feature, relu=False)
        teacher_features_ff = self.ff_tt(g_t[0][0], relu=False)
        # loss_t = self.kullback_leibler_loss(student_feature_ff, teacher_features_ff)
        loss_t = F.mse_loss(student_feature_ff, teacher_features_ff)

        student_feature = student_feature.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        student_feature = self.attn_f(student_feature)

        student_feature_ff = self.ff_fs(student_feature, relu=False)
        teacher_features_ff = self.ff_ft(g_t[0][1], relu=False)
        # loss_f = self.kullback_leibler_loss(student_feature_ff, teacher_features_ff)
        loss_f = F.mse_loss(student_feature_ff, teacher_features_ff)

        return loss_t + loss_f