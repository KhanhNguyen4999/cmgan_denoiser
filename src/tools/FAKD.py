import torch.nn as nn
import torch.nn.functional as F
import torch
        
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
    def __init__(self, args):
        super(FAKD, self).__init__()
        self.n_layer = len(args.t_shapes) # number of student layer and teacher layer should be equal
        
        self.ABF_blocks = [ABF(
                                in_channel = s_shape[1], 
                                mid_channel = t_shape[1], 
                                out_channel = t_shape[1],
                                fuse = i!=self.n_layer-1,
                                H = t_shape[2],
                                W = t_shape[3]
                            ).cuda() 
                            for i, (s_shape, t_shape) in enumerate(zip(args.s_shapes, args.t_shapes))
                        ]

        self.channel_wise_operation_student = nn.ModuleList([nn.Conv2d(t_shape[1], 1, (3,3), padding=1) for t_shape in args.t_shapes])
        self.channel_wise_operation_teacher = nn.ModuleList([nn.Conv2d(t_shape[1], 1, (3,3), padding=1) for t_shape in args.t_shapes])

    def fusion_loss(self, student_layer, teacher_layer, n):
        if n == self.n_layer - 1:
            fusion_feature, _ = self.ABF_blocks[n](student_layer[n])
            fusion_feature = self.channel_wise_operation_student[n](fusion_feature)
            teacher_feature = self.channel_wise_operation_teacher[n](teacher_layer[n])
            loss = F.mse_loss(fusion_feature, teacher_feature)
            return loss, fusion_feature
        
        loss, residual_feature = self.fusion_loss(student_layer, teacher_layer, n+1)
        curr_fusion_features, _ = self.ABF_blocks[n](student_layer[n], residual_feature) 

        fusion_feature = self.channel_wise_operation_student[n](curr_fusion_features)
        teacher_feature = self.channel_wise_operation_teacher[n](teacher_layer[n])
        curr_loss = F.mse_loss(fusion_feature, teacher_feature)
        return loss + curr_loss,  curr_fusion_features
    
    def forward(self, g_s, g_t):
        # Calculate in the frequency site first
        loss, _ = self.fusion_loss(g_s, g_t, 0)

        return loss