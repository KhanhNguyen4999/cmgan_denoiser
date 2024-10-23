import numpy as np
from joblib import Parallel, delayed
from pesq import pesq
from tools.compute_metrics import stoi
from utils import *


def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


# def batch_pesq(clean, noisy, n_jobs=None):
#     if n_jobs == None:
#         n_jobs = -1
#     pesq_score = Parallel(n_jobs=n_jobs)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
#     pesq_score = np.array(pesq_score)
#     if -1 in pesq_score:
#         print("PESQ: ", pesq_score)
#         tmp = pesq_score[pesq_score!=-1]
#         tmp = np.mean(tmp)
#         pesq_score = np.where(pesq_score==-1, tmp, pesq_score)
        
#     pesq_score = (pesq_score - 1) / 3.5
#     return torch.FloatTensor(pesq_score).to('cuda')

def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(
        delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy)
    )
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to("cuda")

def batch_stoi(clean, noisy, n_jobs):
    stoi_score = Parallel(n_jobs=n_jobs*2)(delayed(stoi)(c, n, 16000) for c, n in zip(clean, noisy))
    stoi_score = np.array(stoi_score)
        
    return torch.FloatTensor(stoi_score).to('cuda')

class Discriminator(nn.Module):
    def __init__(self, ndf, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.PReLU(2*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.PReLU(8*ndf),
            # nn.AdaptiveMaxPool2d(1),
            nn.AvgPool2d(kernel_size=(12, 20)),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4)),
            nn.Dropout(0.3),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Linear(ndf*4, 1)),
            LearnableSigmoid(1)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)


class SNRMixingDiscriminator(nn.Module):
    def __init__(self, ndf, in_channel=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.PReLU(2*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.PReLU(8*ndf),
            # nn.AdaptiveMaxPool2d(1),
            nn.AvgPool2d(kernel_size=(12, 20)),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4)),
            nn.Dropout(0.3),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Linear(ndf*4, 1)),
            LearnableSigmoid(1)
        )

    def forward(self, x):
        return self.layers(x)