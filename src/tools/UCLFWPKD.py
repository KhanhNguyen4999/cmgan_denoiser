import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from geomloss import SamplesLoss

class ABF_Res(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF_Res, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            # nn.BatchNorm2d(mid_channel),
            nn.InstanceNorm2d(mid_channel, affine=True),
            nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU()
        )

        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None


    def forward(self, x, y=None, shape = None):
        n, c, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, shape, mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * (z[:,0].view(n,1,h,w).contiguous()) + y * (z[:,1].view(n,1,h,w).contiguous()))
        # output
        y = self.conv2(x)
        return y, x

class ResKD(nn.Module):
    def __init__(
        self, in_channels, out_channels, mid_channel, shapes
    ):
        super(ResKD, self).__init__()
        self.shapes = shapes

        abfs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF_Res(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))


        self.abfs = abfs[::-1]

    def forward(self, student_features):

        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for features, abf, shape in zip(x[1:], self.abfs[1:], self.shapes[1:]):
            out_features, res_features = abf(features, res_features, shape)
            results.insert(0, out_features)

        return results


def cosine_pairwise_similarities_perframe(features, eps=1e-6, normalized=True):
    features = features.permute(1, 0, 2)  ##### take timesteps as the first dim
    features_norm = torch.sqrt(torch.sum(features ** 2, dim=2, keepdim=True))
    features = features / (features_norm + eps)
    # features[features != features] = 0
    features_t = features.permute(0, 2, 1)
    similarities = torch.bmm(features, features_t)

    if normalized:
        similarities = (similarities + 1.0) / 2.0
    return similarities

def pairwise_distances(a, b=None, eps=1e-6):
    """
    Calculates the pairwise distances between matrices a and b (or a and a, if b is not set)
    
    :param a: Tensor of shape (batch_size, time_step, D)
    :param b: Optional Tensor of shape (batch_size, time_step, D)
    :param eps: Small value to ensure numerical stability
    :return: Tensor of shape (batch_size, time_step, time_step) with pairwise distances
    """
    if b is None:
        b = a

    a = a.permute(1, 0, 2)  ##### take timesteps as the first dim
    b = b.permute(1, 0, 2)  ##### take timesteps as the first dim

    aa = torch.sum(a ** 2, dim=-1, keepdim=True)  # Shape: (time_step, batch_size, 1)
    bb = torch.sum(b ** 2, dim=-1, keepdim=True)  # Shape: (time_step, batch_size, 1)
    
    ab = torch.matmul(a, b.transpose(-2, -1))  # Shape: (time_step, batch_size, batch_size)
    
    dists = aa + bb.transpose(-2, -1) - 2 * ab  # Broadcasting to get the shape: (time_step, batch_size, batch_size)
    dists = torch.clamp(dists, min=eps)  # Ensuring no negative distances due to floating point errors
    dists = torch.sqrt(dists)  # Euclidean distance
    return dists

def t_student_kernel_loss_perframe(teacher_features, student_features):

    d = 1
    teacher_d = pairwise_distances(teacher_features)
    teacher_s = 1.0 / (1 + teacher_d ** d)

    student_d = pairwise_distances(student_features)
    student_s = 1.0 / (1 + student_d ** d)

    # Transform them into probabilities
    teacher_s = teacher_s / torch.sum(teacher_s, dim=2, keepdim=True)
    student_s = student_s / torch.sum(student_s, dim=2, keepdim=True)

    loss = (teacher_s - student_s) * (torch.log(teacher_s) - torch.log(student_s))
    
    return loss

def probabilistic_loss_perframe_withFW(teacher_features, student_features, eps=1e-6):

    student_s = cosine_pairwise_similarities_perframe(student_features)
    teacher_s = cosine_pairwise_similarities_perframe(teacher_features)

    # Transform them into probabilities
    teacher_s = teacher_s / torch.sum(teacher_s, dim=2, keepdim=True)
    student_s = student_s / torch.sum(student_s, dim=2, keepdim=True)

    # loss = (teacher_s - student_s) * (torch.log(teacher_s) - torch.log(student_s)) * tea_FW_mask
    loss = (teacher_s - student_s) * (torch.log(teacher_s) - torch.log(student_s))
    # loss = (torch.mean(loss, dim=[1,2])).sum(0)

    return loss


def wasserstein_distance_withFW(teacher_features, student_features):
    # teacher_features = teacher_features.permute(1, 0, 2).contiguous()  ##### take timesteps as the first dim
    # student_features = student_features.permute(1, 0, 2).contiguous()  ##### take timesteps as the first dim

    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=1.0, scaling=0.9, debias=True)
    loss = sinkhorn(teacher_features, student_features) # (time_step, batch_size, C*F) -> (timestep)
    return loss


def combine_loss(teacher_features, student_features, eps=1e-6):
    tea_avg = torch.mean(teacher_features, dim=2, keepdim=True)
    tea_FW_mask = torch.sigmoid(tea_avg).permute(1,0,2)
    
    cosine_loss = probabilistic_loss_perframe_withFW(teacher_features, student_features) * tea_FW_mask
    # t_student_loss = t_student_kernel_loss_perframe(teacher_features, student_features) * tea_FW_mask
    # wasserstein_loss = wasserstein_distance_withFW(teacher_features, student_features)
    # loss = (torch.mean(cosine_loss, dim=[1,2])).sum(0) + (torch.mean(t_student_loss, dim=[1,2])).sum(0) + wasserstein_loss.mean() * 0.001
    loss = (torch.mean(cosine_loss, dim=[1,2])).sum(0)

    return loss

class UCLFWPKD(nn.Module):
    def __init__(self, t_shapes_enc, s_shapes_enc, t_shapes_dec, s_shapes_dec):
        super(UCLFWPKD, self).__init__()
        mid_channel = 128
        in_channels_enc = [s_shape[1] for s_shape in s_shapes_enc]
        out_channels_enc = [t_shape[1] for t_shape in t_shapes_enc]
        shapes_enc = [(t_shape[2], t_shape[3]) for t_shape in s_shapes_enc][::-1]
        self.Review_KD_Block_enc = ResKD(in_channels_enc, out_channels_enc, mid_channel, shapes_enc)

        in_channels_dec = [s_shape[1] for s_shape in s_shapes_dec][::-1]
        out_channels_dec = [t_shape[1] for t_shape in t_shapes_dec][::-1]
        shapes_dec = [(t_shape[2], t_shape[3]) for t_shape in s_shapes_dec]
        self.Review_KD_Block_dec = ResKD(in_channels_dec, out_channels_dec, mid_channel, shapes_dec)

    def forward(self, enc_stu_fea_list, enc_tea_fea_list, dec_stu_fea_list, dec_tea_fea_list):
        pred_res_enc_loss = 0.0
        enc_stu_res_list = self.Review_KD_Block_enc(enc_stu_fea_list)
        # print("Start: ", pred_res_enc_loss)
        for fs, ft in zip(enc_stu_res_list, enc_tea_fea_list):
            BS, CS, T, DS = fs.shape
            BT, CT, T, DT = ft.shape
            ft = ft.detach()

            fs_fea = fs.permute(0, 2, 1, 3)
            ft_fea = ft.permute(0, 2, 1, 3)
            fs_fea = torch.reshape(fs_fea, [BS, T, CS * DS])
            ft_fea = torch.reshape(ft_fea, [BT, T, CT * DT])
            pred_res_enc_loss += combine_loss(ft_fea, fs_fea)
            

        # print("Middle: ",pred_res_enc_loss)
        ###### KD for Dec
        dec_stu_fea_list = dec_stu_fea_list[::-1]
        dec_stu_res_list = self.Review_KD_Block_dec(dec_stu_fea_list)
        dec_stu_res_list = dec_stu_res_list[::-1]
        pred_res_dec_loss = 0.0
        for fs, ft in zip(dec_stu_res_list, dec_tea_fea_list):
            BS, CS, T, DS = fs.shape
            BT, CT, T, DT = ft.shape
            ft = ft.detach()

            fs_fea = fs.permute(0, 2, 1, 3)
            ft_fea = ft.permute(0, 2, 1, 3)
            fs_fea = torch.reshape(fs_fea, [BS, T, CS * DS])
            ft_fea = torch.reshape(ft_fea, [BT, T, CT * DT])
            # dec_loss = probabilistic_loss_perframe_withFW(ft_fea, fs_fea) + t_student_kernel_loss_perframe(ft_fea, fs_fea) + wasserstein_distance_withFW(ft_fea, fs_fea)
            pred_res_dec_loss += combine_loss(ft_fea, fs_fea)
            # print("3: ", dec_loss)
            # pred_res_dec_loss += t_student_kernel_loss_perframe(ft_fea, fs_fea)
            # print("4: ", pred_res_dec_loss)
            
        # print("----Final dec: ", pred_res_dec_loss)
        KD_loss = pred_res_enc_loss + pred_res_dec_loss
        print(KD_loss)
        return KD_loss


if __name__ == '__main__':
    batch_size = 4    
    s_shapes_enc = [(batch_size, 16, 321, 201), (batch_size, 32, 160, 100), (batch_size, 64, 80, 50), (batch_size, 128, 40, 25), (batch_size, 128, 20, 12)]
    t_shapes_enc = [(batch_size, 64, 321, 201), (batch_size, 128, 160, 100), (batch_size, 256, 80, 50), (batch_size, 512, 40, 25), (batch_size, 512, 20, 12)]
    s_shapes_dec = [(batch_size, 64, 40, 25), (batch_size, 32, 80, 50), (batch_size, 16, 160, 100), (batch_size, 16, 321, 201)]
    t_shapes_dec = [(batch_size, 256, 40, 25), (batch_size, 128, 80, 50), (batch_size, 64, 160, 100), (batch_size, 64, 321, 201)]

    enc_stu_fea_list = []
    stu_enc_layer_1_out = torch.randn(4, 16, 321, 201)
    enc_stu_fea_list.append(stu_enc_layer_1_out)
    stu_enc_layer_2_out = torch.randn(4, 32, 160, 100)
    enc_stu_fea_list.append(stu_enc_layer_2_out)
    stu_enc_layer_3_out = torch.randn(4, 64, 80, 50)
    enc_stu_fea_list.append(stu_enc_layer_3_out)
    stu_enc_layer_4_out = torch.randn(4, 128, 40, 25)
    enc_stu_fea_list.append(stu_enc_layer_4_out)
    stu_enc_layer_5_out = torch.randn(4, 128, 20, 12)
    enc_stu_fea_list.append(stu_enc_layer_5_out)


    dec_stu_fea_list = []
    stu_dec_layer_1_out = torch.randn(4, 64, 40, 25)
    dec_stu_fea_list.append(stu_dec_layer_1_out)
    stu_dec_layer_2_out = torch.randn(4, 32, 80, 50)
    dec_stu_fea_list.append(stu_dec_layer_2_out)
    stu_dec_layer_3_out = torch.randn(4, 16, 160, 100)
    dec_stu_fea_list.append(stu_dec_layer_3_out)
    stu_dec_layer_4_out = torch.randn(4, 16, 321, 201)

    enc_tea_fea_list = []
    tea_enc_layer_1_out = torch.randn(4, 64, 321, 201)
    enc_tea_fea_list.append(tea_enc_layer_1_out)
    tea_enc_layer_2_out = torch.randn(4, 128, 160, 100)
    enc_tea_fea_list.append(tea_enc_layer_2_out)
    tea_enc_layer_3_out = torch.randn(4, 256, 80, 50)
    enc_tea_fea_list.append(tea_enc_layer_3_out)
    tea_enc_layer_4_out = torch.randn(4, 512, 40, 25)
    enc_tea_fea_list.append(tea_enc_layer_4_out)
    tea_enc_layer_5_out = torch.randn(4, 512, 20, 12)
    enc_tea_fea_list.append(tea_enc_layer_5_out)

    dec_tea_fea_list = []
    tea_dec_layer_1_out = torch.randn(4, 256, 40, 25)
    dec_tea_fea_list.append(tea_dec_layer_1_out)
    tea_dec_layer_2_out = torch.randn(4, 128, 80, 50)
    dec_tea_fea_list.append(tea_dec_layer_2_out)
    tea_dec_layer_3_out = torch.randn(4, 64, 160, 100)
    dec_tea_fea_list.append(tea_dec_layer_3_out)
    tea_dec_layer_4_out = torch.randn(4, 64, 321, 201)

    distiller = UCLFWPKD(t_shapes_enc, s_shapes_enc, t_shapes_dec, s_shapes_dec)

    loss = distiller(enc_stu_fea_list, enc_tea_fea_list, dec_stu_fea_list, dec_tea_fea_list)
    print("Loss: ", loss)
    # Define two sets of samples (distributions)
    # x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], requires_grad=True)  # Source distribution
    # y = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], requires_grad=True)  # Target distribution
    # Initialize Sinkhorn distance with epsilon and max_iter parameters
    # sinkhorn = SamplesLoss("sinkhorn", p=2, blur=0.1, scaling=0.9, debias=True)
    # x = torch.randn(4, 256, 1000)
    # y = torch.randn(4, 256, 1000)
    # Compute the Sinkhorn distance
    # distance = sinkhorn(x, y)

    # print(f"Sinkhorn distance: {distance}")


