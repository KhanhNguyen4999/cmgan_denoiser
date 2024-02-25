import torch.nn as nn
import torch.nn.functional as F
import torch


class LinearTransformStudent(nn.Module):
    def __init__(self, channel_in):
        super(LinearTransformStudent, self).__init__()

        self.student_channel_wise = nn.Conv2d(channel_in, 3, 1)
        self.fc_block = nn.Sequential(
            nn.Linear(channel_in, 256),
            nn.ReLU()
        )
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_wise_pooling = nn.Conv2d(channel_in, 1, 1)
        self.up = torch.nn.Upsample(size=(321, 101), mode='bilinear', align_corners=False)

    def forward(self, feature):
        # (b, c, t, f) - > (b, c, 1, 1)
        h_s = self.pooling(feature)
        # (b, c, 1, 1) -> (b, 1, c)
        b, c, _, _ = h_s.size()
        h_s = h_s.permute(0, 2, 3, 1).contiguous().view(b, 1, c)
        # (b, 1, c) -> (b, 1, 256)
        h_s = self.fc_block(h_s)

        # upsampling (b, c, student_t, student_f) -> (b, c, teacher_t, teacher_f)
        features_upsampling = self.up(feature)
        # (b, 1, teacher_t, teacher_f)
        feature_channel_wise = self.channel_wise_pooling(features_upsampling)
        return h_s, feature_channel_wise


        

class LinearTransformTeacher(nn.Module):
    def __init__(self, channel_in):
        super(LinearTransformTeacher, self).__init__()

        self.teacher_fc_block = nn.Sequential(
            nn.Linear(channel_in, 256),
            nn.ReLU()
        )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.teacher_channel_wise = nn.Conv2d(channel_in, 1, 1)

    def forward(self, feature):
        b, c, t, f = feature.size()
        # (b, c, t, f) -> (b, c, 1, 1)
        h_t = self.pooling(feature)
        # (b, c, 1, 1) -> (b, 1, c)
        h_t = h_t.permute(0, 2, 3, 1).contiguous().view(b, 1, c)
        # (b, 1, c) -> (b, 1, 256)
        h_t = self.teacher_fc_block(h_t)
        # (b, c, t, f) -> (b, 1, t, f)
        teacher_feature_channel_wise = self.teacher_channel_wise(feature)

        return h_t, teacher_feature_channel_wise
    

class FKD(nn.Module):
    def __init__(self):
        super(FKD, self).__init__()
        self.linear_transform_s_block = nn.ModuleList([
            LinearTransformStudent(256),
            LinearTransformStudent(128),
            LinearTransformStudent(64),
            LinearTransformStudent(32)
        ])
        self.linear_transform_t = LinearTransformTeacher(64)
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, teacher_feature, student_features):
        """
        t_feature: tensor (b, c, t, f)
        s_features: list of tensor [(b, c1, t1, f1), (b, c2, t2, f2), ...]
        """

        b, c, t, f = teacher_feature.size()
        h_t, teacher_feature_channel_wise = self.linear_transform_t(teacher_feature)

        h_s_list = []
        loss_list = []
        for i in range(len(student_features)):
            h_s, student_feature_channel_wise = self.linear_transform_s_block[i](student_features[i])
            h_s_list.append(h_s)

            # out: (b, c * teacher_t * teacher_f)
            loss = self.mse_loss(student_feature_channel_wise.view(b, -1), teacher_feature_channel_wise.view(b, -1))
            loss = loss.mean(1)
            loss_list.append(loss)

        # out: (b, len(student_feature), 256)
        h_s_tensor = torch.cat(h_s_list, dim=1)
        # (b, 4, 256) x (b, 1, 256) = (b, 4, 1)
        weight = torch.matmul(h_s_tensor, h_t.transpose(1,2))
        weight = weight.squeeze()
        weight_softmax = torch.softmax(weight, dim=1)

        # out: (b, 4)
        loss_by_encoder_features = torch.stack(loss_list, dim=1)
        # out: (b)
        loss_by_encoder_features = (loss_by_encoder_features * weight_softmax).sum(1)

        return loss_by_encoder_features