import torch.nn as nn
import torch
from s3prl.nn import S3PRLUpstream

class ContrastiveLoss(nn.Module):
    def __init__(self, model_name):
        self.model = S3PRLUpstream(model_name) 
        self.model.eval()
        self.euclidean_distance = nn.PairwiseDistance(p=2)

    def forward(self, noisy_wav, clean_wav, enhance_wav):
        batch_size = noisy_wav.shape[0]
        x = torch.concat(noisy_wav, clean_wav, enhance_wav)
        with torch.no_grad():
            emb = self.model(x)
            emb = emb[0].mean(1)

        noisy_wav_emb = emb[:batch_size, :]
        clean_wav_emb = emb[batch_size:2*batch_size, :]        
        enhance_wav_emb = emb[2*batch_size:, :]  

        loss = self.euclidean_distance(clean_wav_emb, enhance_wav_emb) / self.euclidean_distance(noisy_wav_emb, enhance_wav_emb) 
        loss = loss.mean()
        return loss






if __name__ == "__main__":
    None
    # 