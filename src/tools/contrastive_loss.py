import torch.nn as nn
import torch
from s3prl.nn import S3PRLUpstream

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

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.linear = nn_bn_relu(768, 512)

    def forward(self, noisy_wav, clean_wav, enhance_wav, teacher_enhance_wav):
    
        clean_enhance_dist = torch.mean(torch.abs(clean_wav - enhance_wav))
        teacher_enhance_dist = torch.mean(torch.abs(teacher_enhance_wav - enhance_wav))
        noisy_enhance_dist = torch.mean(torch.abs(noisy_wav - enhance_wav))

        loss = (teacher_enhance_dist + clean_enhance_dist) / noisy_enhance_dist
        return loss






if __name__ == "__main__":
    None
    # 