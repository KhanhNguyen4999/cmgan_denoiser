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
    
    # def forward(self, student_feature, teacher_feature):
    #     loss = F.mse_loss(student_feature, teacher_feature)
    #     return loss
    def forward(self, student_feature, teacher_feature):
        b, c, t, f = student_feature.size()
        student_feature_t = student_feature.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        teacher_feature_t = teacher_feature.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        student_feature_t = self.attn_t(student_feature_t)
        student_s1 = cosine_pairwise_similarities(student_feature_t, normalized=True) # normalize to avoid the negative value, it would lead the nan value in log function
        teacher_s1 = cosine_pairwise_similarities(teacher_feature_t, normalized=True)
        student_s1 = F.softmax(student_s1, dim=2)
        teacher_s1 = F.softmax(teacher_s1, dim=2)
        loss_t = (teacher_s1 - student_s1) * (torch.log(teacher_s1) - torch.log(student_s1))
        loss_t = loss_t.sum(dim=1).sum(dim=1)


        student_feature_f = student_feature.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)
        teacher_feature_f = teacher_feature.permute(0, 2, 3, 1).contiguous().view(b*t, f, c)
        student_feature_f = self.attn_f(student_feature_f)
        student_s2 = cosine_pairwise_similarities(student_feature_f, normalized=True) # normalize to avoid the negative value, it would lead the nan value in log function
        teacher_s2 = cosine_pairwise_similarities(teacher_feature_f, normalized=True)
        student_s2 = F.softmax(student_s2, dim=2)
        teacher_s2 = F.softmax(teacher_s2, dim=2)
        loss_f = (teacher_s2 - student_s2) * (torch.log(teacher_s2) - torch.log(student_s2))
        loss_f = loss_f.sum(dim=1).sum(dim=1)
        
        return torch.mean(loss_t) + torch.mean(loss_f)

class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse, H, W):
        super(ABF, self).__init__()
        self.shape = (H, W)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(), # có nên đổi thành relu không
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None):
        '''
            x: current student layer feature
            y: residual feature
        '''
        # upsample student feature
        x = F.interpolate(x, self.shape, mode="nearest") # or do upsampling
        n,_,h,w = x.shape
        x = self.conv1(x)
        if self.att_conv is not None:
            # fusion student feature and residual feature
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output 
        y = self.conv2(x)
        return y, x

class FAKD(nn.Module):
    def __init__(self, s_shapes, t_shapes):
        super(FAKD, self).__init__()
        self.linear_trans_s = LinearTransformStudent(s_shapes, t_shapes)  # Ensure this class is defined
        self.mse_loss = [MSELoss(t_shape[1]) for t_shape in t_shapes]
        self.n_layer = len(s_shapes)

        self.ABF_blocks = [ABF(
                                in_channel = s_shape[1], 
                                mid_channel = t_shape[1], 
                                out_channel = t_shape[1],
                                fuse = i!=self.n_layer-1,
                                H = t_shape[2],
                                W = t_shape[3]
                            ).cuda() 
                            for i, (s_shape, t_shape) in enumerate(zip(s_shapes, t_shapes))]

    def fusion_loss(self, student_layer, teacher_layer, n=0):
        if n == self.n_layer - 1:
            fusion_feature, _ = self.ABF_blocks[n](student_layer[n])
            loss = self.mse_loss[n](fusion_feature, teacher_layer[n])
            return loss, fusion_feature
        
        loss, residual_feature = self.fusion_loss(student_layer, teacher_layer, n+1)
        curr_fusion_features, _ = self.ABF_blocks[n](student_layer[n], residual_feature) 
        curr_loss = self.mse_loss[n](curr_fusion_features, teacher_layer[n])
        return loss + curr_loss,  curr_fusion_features
    
    def forward(self, student_feats, teacher_feats):
        # Calculate in the frequency site first
        loss, _ = self.fusion_loss(student_feats, teacher_feats)
        return loss



    