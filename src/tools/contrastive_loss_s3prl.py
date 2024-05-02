import torch.nn as nn
import torch.nn.functional as F
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
    
class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class ContrastiveLoss(nn.Module):
    def __init__(self, model_name):
        super(ContrastiveLoss, self).__init__()
        self.model = S3PRLUpstream(model_name)
        self.model.eval()
        self.linear_noisy = Embed(768, 512)
        self.linear_clean = Embed(768, 512)
        self.linear_enhance = Embed(768, 512)
        self.linear_teacher_enhance = Embed(768, 512)
        self.channel_wise = nn.ModuleList([torch.nn.Conv1d(100, 1, 3, padding=1) for _ in range(4)])

    def forward(self, noisy_wav, clean_wav, enhance_wav, teacher_enhance_wav):
        batch_size = noisy_wav.shape[0]
    
        wavs_len = torch.LongTensor([16000 * 2] * batch_size * 4)
        x = torch.concat([noisy_wav, clean_wav, enhance_wav, teacher_enhance_wav])

        # with torch.set_grad_enabled(True):
        emb, _ = self.model(x, wavs_len)
        emb = emb[0]

        noisy_wav_emb = self.channel_wise[0](emb[:batch_size]).squeeze()
        clean_wav_emb = self.channel_wise[1](emb[batch_size:2*batch_size]).squeeze()      
        enhance_wav_emb = self.channel_wise[2](emb[2*batch_size:3*batch_size]).squeeze()  
        teacher_enhance_wav_emb = self.channel_wise[3](emb[3*batch_size:]).squeeze()

        noisy_wav_emb = self.linear_noisy(noisy_wav_emb)
        clean_wav_emb = self.linear_clean(clean_wav_emb)
        enhance_wav_emb = self.linear_enhance(enhance_wav_emb)
        teacher_enhance_wav_emb = self.linear_teacher_enhance(teacher_enhance_wav_emb)

        # loss = (self.euclidean_distance(clean_wav_emb, enhance_wav_emb)) / self.euclidean_distance(noisy_wav_emb, enhance_wav_emb) 
        clean_enhance_dist = torch.nn.functional.pdist(clean_wav_emb - enhance_wav_emb)
        teacher_enhance_dist = torch.nn.functional.pdist(teacher_enhance_wav_emb - enhance_wav_emb)
        noisy_enhance_dist = torch.nn.functional.pdist(noisy_wav_emb - enhance_wav_emb)

        loss = (clean_enhance_dist) / (noisy_enhance_dist + teacher_enhance_dist)
        # print("--------------")
        # print("clean enhance dist: ", clean_enhance_dist.mean())
        # print("teacher enhance dist: ", teacher_enhance_dist.mean())
        # print("noisy enhance dist: ", noisy_enhance_dist.mean())

        loss = loss.sum()/loss.size(0)
        return loss






if __name__ == "__main__":
    None
    # 