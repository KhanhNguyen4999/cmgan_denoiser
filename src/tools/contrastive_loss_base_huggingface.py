import torch.nn as nn
import torch
from transformers import AutoProcessor, WavLMModel
import torch


class ContrastiveLoss(nn.Module):
    def __init__(self, model_name):
        super(ContrastiveLoss, self).__init__()
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.eval()
        self.model.feature_grad_mult = 1.0
        self.euclidean_distance = nn.PairwiseDistance(p=2)
        self.sampling_rate = 16000

    def forward(self, noisy_wav, clean_wav, enhance_wav, teacher_enhance_wav):
        batch_size = noisy_wav.shape[0]
    
        x = torch.concat([noisy_wav, clean_wav, enhance_wav, teacher_enhance_wav])   

        inputs = {"input_values": x.cuda(), 
                    "attention_mask": torch.ones(batch_size * 4, self.sampling_rate).to(torch.int32).cuda()}
        
        outputs = self.model(**inputs)

        last_hidden_states_emb = outputs.last_hidden_state.mean(1)

        noisy_wav_emb = last_hidden_states_emb[:batch_size, :]
        clean_wav_emb = last_hidden_states_emb[batch_size:2*batch_size, :]        
        enhance_wav_emb = last_hidden_states_emb[2*batch_size:3*batch_size, :]  
        teacher_enhance_wav_emb = last_hidden_states_emb[3*batch_size:, :]  

        clean_enhance_dist = torch.nn.functional.pdist(clean_wav_emb - enhance_wav_emb)
        teacher_enhance_dist = torch.nn.functional.pdist(teacher_enhance_wav_emb - enhance_wav_emb)
        noisy_enhance_dist = torch.nn.functional.pdist(noisy_wav_emb - enhance_wav_emb)
        # clean_enhance_dist = self.euclidean_distance(clean_wav_emb, enhance_wav_emb)
        # noisy_enhance_dist = self.euclidean_distance(noisy_wav_emb, enhance_wav_emb) 

        loss = (clean_enhance_dist + teacher_enhance_dist) / noisy_enhance_dist
        
        loss = loss.mean()
        return loss






if __name__ == "__main__":
    None
    # 